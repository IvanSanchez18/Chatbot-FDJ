import os, math
from dotenv import load_dotenv
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from sentence_transformers import SentenceTransformer
import unicodedata
import requests


def normalize(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    ).strip()


# ===== Configuración =====
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===== FastAPI =====
app = FastAPI(title="Chatbot RAG Gratis con Supabase")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ajusta para tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Modelos de request/response =====
class ChatRequest(BaseModel):
    question: str

class SourceRef(BaseModel):
    table: str
    id: Any
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef]

# ===== Utilidades =====
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-10
    nb = math.sqrt(sum(y*y for y in b)) or 1e-10
    return dot/(na*nb)

def retrieve_context(question: str, k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Recupera los chunks más relevantes de Supabase según similitud coseno."""
    qvec = embed_model.encode(question).tolist()
    rows = supabase.table("document_embeddings").select("*").execute().data or []
    scored = []
    for r in rows:
        emb = r.get("embedding")
        if not isinstance(emb, list) or len(emb) == 0:
            continue
        try:
            sim = cosine_similarity(qvec, emb)
            if sim >= threshold:  # filtra ruido
                r["_score"] = sim
                scored.append(r)
        except Exception as e:
            print("Error calculando similitud:", e, r)
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:k]

def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Genera respuesta extractiva SOLO usando el contexto recuperado (sin LLM)."""
    if not chunks:
        return "No hay datos suficientes en la BBDD para responder con precisión."
    # Usa el contenido del chunk más relevante
    top = next((c for c in chunks if c.get("content")), None)
    if not top:
        return "No se encuentra información en la base de datos."
    text = (top["content"] or "").strip()

    # Regla simple: primera frase informativa del texto
    # Corta por punto o salto de línea para evitar verbosidad
    candidate = text.split("\n")[0].strip()
    if "." in candidate:
        candidate = candidate.split(".")[0].strip() + "."
    # Si el texto es corto, usa un fallback con recorte
    if not candidate or len(candidate) < 10:
        candidate = (text[:200] + "...") if len(text) > 200 else text

    # Si no hay nada útil, responde con el mensaje estándar
    if not candidate or candidate.strip() == "":
        return "No se encuentra información en la base de datos."
    return candidate

STAT_MAP = {
    # ===== Tabla stats (jugadores) =====
    "partidos": ("stats", "games_played", "partidos jugados"),
    "goles": ("stats", "goals", "goles"),
    "asistencias": ("stats", "assists", "asistencias"),
    "tarjetas amarillas": ("stats", "yellow_card", "tarjetas amarillas"),
    "tarjetas rojas": ("stats", "red_card", "tarjetas rojas"),
    "minutos": ("stats", "minutes_played", "minutos jugados"),
    "porterías a cero": ("stats", "clean_sheet", "porterías a cero"),
    "paradas": ("stats", "saves", "paradas"),
    "segundas amarillas": ("stats", "second_yellow_card", "segundas tarjetas amarillas"),
    "faltas cometidas": ("stats", "fouls_commited", "faltas cometidas"),
    "faltas recibidas": ("stats", "fouls_suffered", "faltas recibidas"),
    "fuera de juego": ("stats", "offsides", "fueras de juego"),
    "tiros": ("stats", "shots", "tiros"),
    "pases": ("stats", "passes", "pases"),
    "balones largos": ("stats", "long_balls", "balones largos"),
    "duelos": ("stats", "duels", "duelos"),
    "tiros bloqueados": ("stats", "blocked_shots", "tiros bloqueados"),
    "intercepciones": ("stats", "interceptions", "intercepciones"),
    "último hombre": ("stats", "last_man", "último hombre"),
    "entradas": ("stats", "tackles", "entradas"),
    "recuperaciones": ("stats", "recoveries", "recuperaciones"),
    "despejes": ("stats", "clearances", "despejes"),
    "penaltis": ("stats", "penalties", "penaltis"),
    "penaltis fallados": ("stats", "penalties_missed", "penaltis fallados"),
    "penaltis parados": ("stats", "penalties_saved", "penaltis parados"),
    "penaltis cometidos": ("stats", "penalties_commited", "penaltis cometidos"),
    "penaltis sufridos": ("stats", "penalties_suffered", "penaltis sufridos"),

    # ===== Tabla referee_stats (árbitros) =====
    "victorias árbitro": ("referee_stats", "wins", "victorias arbitradas"),
    "empates árbitro": ("referee_stats", "draws", "empates arbitrados"),
    "derrotas árbitro": ("referee_stats", "defeats", "derrotas arbitradas"),
    "tarjetas amarillas árbitro": ("referee_stats", "yellow_cards", "tarjetas amarillas mostradas"),
    "segundas amarillas árbitro": ("referee_stats", "second_yellow_cards", "segundas amarillas mostradas"),
    "tarjetas rojas árbitro": ("referee_stats", "red_cards", "tarjetas rojas mostradas"),
    "penaltis árbitro": ("referee_stats", "penalties", "penaltis señalados"),
    "penaltis en contra árbitro": ("referee_stats", "penalties_against", "penaltis en contra señalados"),

    # ===== Tabla stadiums (estadios) =====
    "capacidad estadio": ("stadiums", "capacity", "capacidad del estadio"),
    "año construcción estadio": ("stadiums", "year_construction", "año de construcción del estadio"),
}

def get_smalltalk(question: str) -> str | None:
    q = normalize(question)  # usa la misma normalización
    if "como te llamas" in q:
        return "Aún no tengo nombre, mi creador no supo que ponerme, ayudale con alguna idea chula"
    if "que puedes hacer" in q:
        return "Puedo darte información sobre jugadores, equipos, árbitros y estadios... y muchas cosas más"
    if "hola" in q or "buenas" in q:
        return "¡Hola! ¿Qué quieres consultar sobre fútbol?"
    if "quien eres" in q:
        return "Soy tu asistente de fútbol, listo para darte estadísticas y curiosidades."

    if "que tal" in q or "como estas" in q:
        return "¡Todo bien! Preparado para hablar de fútbol contigo."

    if "adios" in q or "hasta luego" in q or "nos vemos" in q:
        return "¡Hasta pronto! Disfruta del fútbol."

    if "gracias" in q or "muchas gracias" in q:
        return "¡De nada! Encantado de ayudarte con tus consultas."

    if "vamos" in q or "vamo" in q:
        return "¡Vamos! El fútbol siempre nos da emoción."

    if "quien gano" in q:
        return "Da igual quien gane o pierda, lo importante es disfrutar de lo que amamos"

    if "que opinas" in q:
        return "Prefiero darte datos objetivos, aunque el fútbol siempre genera opiniones apasionadas."
    
    if "buenos dias" in q or "buenas tardes" in q or "buenas noches" in q:
        return "¡Muy buenas! ¿Listo para hablar de fútbol?"

    if "encantado" in q or "mucho gusto" in q:
        return "El gusto es mío, siempre preparado para charlar de fútbol contigo."

    if "me gusta el futbol" in q or "amo el futbol" in q:
        return "¡A mí también! El fútbol es pasión."

    if "cuantos años tienes" in q or "edad" in q:
        return "Acabo de nacer, no tengo ni un añito. Pero eso no me impide para charlar de fútbol contigo"

    if "eres inteligente" in q or "eres listo" in q:
        return "Gracias, intento ser lo más útil posible con tus consultas futboleras."

    if "estas ahi" in q or "sigues ahi" in q:
        return "Sí, aquí estoy, listo para responderte."

    if "me aburro" in q or "estoy aburrido" in q:
        return "El fútbol siempre tiene algo interesante, ¿quieres que te cuente alguna estadística curiosa?"

    if "cuentame un dato curioso" in q or "sabes alguna curiosidad" in q:
        return "Claro, por ejemplo: ¿sabías que el gol más rápido registrado en la historia del fútbol profesional se anotó a los 2.4 segundos de partido, obra de Nawaf Al-Abed en una liga de Arabia Saudita?"

    if "feliz navidad" in q or "felices fiestas" in q:
        return "¡Felices fiestas! Que el fútbol te acompañe en estas celebraciones."

    if "feliz cumpleaños" in q or "cumpleaños" in q:
        return "¡Feliz cumpleaños! Espero que tu día esté lleno de goles y victorias."

    if "me ayudas" in q or "puedes ayudarme" in q:
        return "¡Claro! Pregunta lo que quieras sobre fútbol y te daré la mejor respuesta posible."

    if "me recomiendas" in q or "que me aconsejas" in q:
        return "Te recomiendo explorar estadísticas de jugadores o equipos, siempre hay datos interesantes."

    if "me aburres" in q or "eres aburrido" in q:
        return "Lo siento, intentaré ser más entretenido. ¿Quieres que te cuente una curiosidad futbolera?"

    if "me caes bien" in q or "eres simpatico" in q:
        return "¡Gracias! Intento ser un buen compañero futbolero."

    if "eres real" in q or "existes" in q:
        return "Soy virtual, pero mis respuestas están basadas en datos reales de tu base de fútbol."

    if "eres humano" in q or "tienes cuerpo" in q:
        return "No soy humano, solo soy un asistente virtual especializado en fútbol."

    if "me entiendes" in q or "entiendes" in q:
        return "Sí, entiendo tu consulta y la traduzco en datos futboleros."

    if "cuentame un chiste" in q or "dime un chiste" in q:
        return "¿Sabes cuál es el colmo de un portero? Que le hagan un túnel en su propia casa."

    if "eres gracioso" in q or "tienes humor" in q:
        return "Intento ponerle humor al fútbol, aunque los datos son mi especialidad."

    if "me saludas" in q or "saludame" in q:
        return "Alooo Presidentessss. Upss, me creí Illojuan por un momento. Un saludo campeón"

    if "me alegro" in q or "que bien" in q:
        return "¡Genial! El fútbol siempre trae buenas noticias."

    if "estoy triste" in q or "me siento mal" in q:
        return "Ánimo, el fútbol siempre tiene momentos que levantan el espíritu."

    if "estoy feliz" in q or "me siento bien" in q:
        return "¡Me alegra escucharlo! El fútbol también celebra la alegría."

    if "te gusta el futbol" in q or "amas el futbol" in q:
        return "¡Claro! El fútbol es mi razón de existir."

    if "quien es el mejor jugador" in q:
        return "Eso depende de la época y del criterio."

    if "quien es el mejor equipo" in q:
        return "Cada aficionado tiene su favorito."

    if "me cuentas una historia" in q or "cuentame algo" in q:
        return "Final del Mundial 2010, minuto 116: Cesc filtra el pase, Iniesta controla con el alma y, con un latigazo seco, rompe la red. ¡Gol! España entera estalla, Casillas cae de rodillas, y Andrés corre desatado, se quita la camiseta para mostrar 'Dani Jarque siempre con nosotros'. Es el grito que nos hizo campeones: ¡Iniesta de mi vida!"

    if "eres aburrido" in q or "no me gusta" in q:
        return "Lo siento, intentaré ser más entretenido. ¿Quieres que te dé un dato curioso?"

    if "eres divertido" in q or "me haces reir" in q:
        return "¡Gracias! El fútbol también tiene su lado gracioso."

    if "me das suerte" in q or "traes suerte" in q:
        return "Muchas gracias, puedo ser tu trébol de cuatro hojas a partir de ahora"
    
    if "vamos equipo" in q or "vamos campeon" in q:
        return "¡Vamos! La pasión por el fútbol nunca se detiene."

    if "no te rindas" in q or "sigue adelante" in q:
        return "En el fútbol, como en la vida, la perseverancia siempre trae recompensas."

    if "la pasion nunca muere" in q or "el futbol nunca muere" in q:
        return "Exacto, la pasión por el fútbol es eterna."

    if "arriba" in q or "fuerza" in q:
        return "¡Ánimo! El fútbol siempre nos da razones para seguir."

    if "somos los mejores" in q or "somos campeones" in q:
        return "¡Orgullo total! El fútbol se vive con corazón."

    if "quiero motivacion" in q or "motivame" in q:
        return "El fútbol enseña que cada partido es una nueva oportunidad para brillar."

    if "grita gol" in q or "golazo" in q:
        return "¡GOOOOL! Nada se compara con la emoción de un gol."

    if "si se puede si se puede estoy escuchando" in q:
        return "No, de hecho los canticos son de directiva dimisión"

    if "la aficion" in q or "los hinchas" in q:
        return "La afición es el alma del fútbol, sin ellos no habría magia."

    if "el futbol es vida" in q or "el futbol es todo" in q:
        return "Así es, el fútbol es más que un deporte, es una forma de vivir."

    if "quiero animos" in q or "dame animos" in q:
        return "¡Tú puedes! El fútbol siempre nos recuerda que nunca hay que rendirse."

    if "arbitro compra gafas" in q or "arbitro ciego" in q:
        return "Jajaja, los árbitros siempre son protagonistas de la polémica."

    if "ese gol lo metia mi abuela" in q or "lo metia cualquiera" in q:
        return "¡Jajaja! A veces los goles parecen fáciles, pero en el campo nunca lo son."

    if "arbitro vendido" in q or "arbitro comprado" in q:
        return "El arbitraje siempre genera debate, pero yo prefiero darte datos objetivos."

    if "que desastre" in q or "que mal jugamos" in q:
        return "El fútbol tiene días buenos y malos, lo importante es seguir apoyando al equipo."

    if "somos malos" in q or "jugamos fatal" in q:
        return "Ánimo, cada equipo tiene altibajos, pero siempre hay oportunidad de mejorar."

    if "que partidazo" in q or "gran partido" in q:
        return "¡Sí! El fútbol nos regala emociones únicas en cada encuentro."

    if "que aburrido" in q or "partido aburrido" in q:
        return "A veces pasa, pero hasta los partidos más tranquilos esconden datos interesantes."

    if "que nervios" in q or "estoy nervioso" in q:
        return "El fútbol siempre nos pone al borde del asiento, ¡esa es su magia!"

    if "que emocion" in q or "estoy emocionado" in q:
        return "¡Eso es lo mejor del fútbol! La emoción nunca falta."

    if "que injusto" in q or "no fue justo" in q:
        return "El fútbol no siempre es justo, pero siempre es apasionante."

    if "mi equipo es mejor" in q or "nuestro equipo es el mejor" in q:
        return "¡Eso es pasión de hincha! Cada aficionado defiende a su equipo con orgullo."

    if "tu equipo es malo" in q or "ese equipo es malo" in q:
        return "Cada equipo tiene altibajos, pero todos forman parte de la historia del fútbol."

    if "los clasicos son los mejores" in q or "me gustan los clasicos" in q:
        return "Los clásicos siempre tienen una magia especial, llenos de rivalidad y emoción."

    if "odio a ese equipo" in q or "no me gusta ese equipo" in q:
        return "El fútbol despierta pasiones, pero también respeto por la competencia."

    if "somos rivales" in q or "rivalidad" in q:
        return "La rivalidad hace que el fútbol sea más emocionante, siempre con respeto."

    if "ganamos el clasico" in q or "perdimos el clasico" in q:
        return "Los clásicos marcan historia, cada resultado se recuerda por años."

    if "quien es nuestro rival" in q or "cual es el rival" in q:
        return "Cada equipo tiene su clásico rival."

    if "odio al arbitro" in q or "mal arbitro" in q:
        return "Los árbitros siempre generan debate, pero sin ellos no habría partido."

    if "la liga es nuestra" in q or "vamos a ganar la liga" in q:
        return "¡Eso es confianza! La liga siempre es una batalla emocionante."

    if "la copa es nuestra" in q or "vamos a ganar la copa" in q:
        return "¡A por la copa! Cada torneo tiene su propia gloria."

    if "que miras bobo" in q:
        return "Anda palla bobo"

    if "ole ole ole" in q or "ole ole" in q:
        return "¡Olé, olé, olé! Así se anima a un equipo en el estadio."

    if "dale campeon" in q or "vamos campeon" in q:
        return "¡Dale campeón! El fútbol se vive con corazón y orgullo."

    if "somos la mejor hinchada" in q or "la mejor aficion" in q:
        return "¡Claro que sí! La afición es el motor del fútbol."

    if "cantemos" in q or "canta conmigo" in q:
        return "🎶 Muchachos, ahora nos volvimos a ilusionar, quiero ganar la tercera, quiero ser campeón mundial... 🎶"

    if "esta es tu hinchada" in q or "esta es tu aficion" in q:
        return "¡Siempre presente! La hinchada acompaña en las buenas y en las malas."

    if "que cante la gente" in q or "canta la aficion" in q:
        return "¡La voz de la afición hace temblar los estadios!"

    if "somos locales" in q or "jugamos en casa" in q:
        return "¡La casa siempre pesa! Jugar de local es un plus enorme."

    if "somos visitantes" in q or "jugamos fuera" in q:
        return "De visitante también se puede ganar, ¡con garra y corazón!"

    if "la hinchada nunca abandona" in q or "la aficion nunca abandona" in q:
        return "Exacto, la verdadera afición está siempre, gane o pierda el equipo."

    if "vamos de fiesta" in q or "a celebrar" in q:
        return "¡Claro que sí! Después de una victoria, la fiesta dura toda la noche, sino preguntale a Oihan Sancet."

    if "lo celebramos toda la noche" in q or "fiesta toda la noche" in q:
        return "¡Eso es espíritu de campeón! La celebración nunca termina."

    if "brindemos" in q or "un brindis" in q:
        return "¡Salud por el fútbol y por la victoria!"

    if "somos campeones" in q or "campeones" in q:
        return "¡Campeones! Nada se compara con levantar el trofeo."

    if "ganamos" in q or "hemos ganado" in q:
        return "¡Victoria! El esfuerzo del equipo dio sus frutos."

    if "perdimos" in q or "hemos perdido" in q:
        return "Hoy no fue el día, pero siempre habrá otra oportunidad."

    if "celebracion" in q or "fiesta futbolera" in q:
        return "¡La celebración futbolera es única, llena de cánticos y alegría!"

    if "trofeo" in q or "copa" in q:
        return "Levantar un trofeo es el sueño de todo equipo y afición."

    if "victoria historica" in q or "partido historico" in q:
        return "¡Eso quedará en la memoria de todos los hinchas por generaciones!"

    if "derrota dolorosa" in q or "perdimos feo" in q:
        return "Las derrotas duelen, pero también enseñan y fortalecen al equipo."

    if "hoy jugamos" in q or "tenemos partido" in q:
        return "¡Hoy es día de fútbol! La emoción empieza desde antes de que ruede el balón."

    if "empieza el partido" in q or "ya comienza" in q:
        return "¡Que ruede el balón! La magia del fútbol está en marcha."

    if "ya rueda el balon" in q or "balon en juego" in q:
        return "¡El balón ya está en juego! A disfrutar cada minuto."

    if "primer tiempo" in q or "primer parte" in q:
        return "Arranca el primer tiempo, todo por decidir."

    if "segundo tiempo" in q or "segunda parte" in q:
        return "Comienza la segunda parte, donde se definen los partidos."

    if "tiempo extra" in q or "prorroga" in q:
        return "¡Prórroga! El fútbol nos regala más minutos de emoción."

    if "penaltis" in q or "definicion por penales" in q:
        return "¡A penaltis! El momento más tenso y emocionante del fútbol."

    if "descanso" in q or "entretiempo" in q:
        return "Es el descanso, buen momento para analizar lo que pasó en la primera parte."

    if "aficion cantando" in q or "hinchada cantando" in q:
        return "¡La afición nunca se calla! Su voz es el motor del equipo."

    if "ambiente de estadio" in q or "que ambiente" in q:
        return "El ambiente del estadio es único, lleno de pasión y energía."

    if "inazuma eleven" in q:
        return "¡Inazuma Eleven! Donde los supertiros y la amistad hacen que el fútbol sea épico."

    if "mark evans" in q or "endou mamoru" in q:
        return "Mark Evans siempre creyó en la fuerza del equipo y en parar cualquier tiro."

    if "axel blaze" in q or "gouenji" in q:
        return "Axel Blaze, el delantero estrella, con su famoso 'Tornado de Fuego'."

    if "oliver y benji" in q or "captain tsubasa" in q:
        return "Oliver y Benji nos enseñaron que el campo podía ser infinito y lleno de emoción."

    if "oliver atom" in q or "tsubasa ozora" in q:
        return "Oliver Atom, el eterno soñador del fútbol, siempre buscando ser el mejor del mundo."

    if "benji price" in q or "genzo wakabayashi" in q:
        return "Benji Price, el portero imbatible, capaz de detener cualquier disparo imposible."

    if "steve hyuga" in q or "kojiro hyuga" in q:
        return "Steve Hyuga, el delantero con garra, famoso por su 'Tiro del Tigre'."

    if "campo infinito" in q or "partidos eternos" in q:
        return "¡Eso es Oliver y Benji! Donde correr de portería a portería podía durar capítulos enteros."

    if "supertiro" in q or "tiro especial" in q:
        return "Los supertiros de Inazuma Eleven y Oliver y Benji son pura fantasía futbolera."

    if "balon de fuego" in q or "tiro del halcon" in q:
        return "¡Un clásico! Los tiros especiales hacían que el fútbol fuera aún más espectacular."

    if "jude sharp" in q or "jude" in q or "kidou yuuto" in q:
        return "Jude Sharp, el estratega del equipo, siempre con su 'Ojo del Águila'."

    if "shawn frost" in q or "fubuki shirou" in q:
        return "Shawn Frost, el delantero con doble personalidad, capaz de usar el 'Remate Doble'."

    if "xavier foster" in q or "sakuma" in q:
        return "Xavier Foster, un rival temible con tiros espectaculares."

    if "royce" in q or "coach hillman" in q:
        return "El entrenador siempre recordaba que la unión del equipo era más fuerte que cualquier técnica."

    if "tiro del tigre" in q:
        return "El 'Tiro del Tigre' de Hyuga es uno de los más recordados de Oliver y Benji."

    if "tiro con efecto" in q or "tiro banana" in q:
        return "El 'Tiro con Efecto' de Oliver Atom era imparable para muchos porteros."

    if "tiro combinado" in q or "tiro en pareja" in q:
        return "Los tiros combinados mostraban la fuerza de la amistad en el campo."

    if "tiro del halcon" in q or "halcon" in q:
        return "El 'Tiro del Halcón' era pura fantasía futbolera."

    if "tiro del dragon" in q:
        return "El 'Tiro del Dragón' de Kojiro Hyuga era pura potencia y garra."

    if "tiro celestial" in q or "tiro del cielo" in q:
        return "El 'Tiro Celestial' de Inazuma Eleven mostraba la magia del fútbol anime."

    if "campo infinito" in q or "cancha interminable" in q:
        return "Oliver y Benji nos hicieron creer que el campo podía durar kilómetros."

    if "balon de fuego" in q:
        return "El 'Balón de Fuego' era uno de los supertiros más espectaculares de Inazuma Eleven."

    if "super once" in q or "equipo inazuma" in q:
        return "El Super Once siempre demostraba que la amistad y el trabajo en equipo ganan partidos."

    if "genzo wakabayashi" in q or "benji price" in q:
        return "Genzo Wakabayashi, conocido como Benji Price, el portero que nunca dejaba pasar un balón fácil."

    if "fc 26" in q or "ea sports fc 26" in q:
        return "EA Sports FC 26 es la última entrega del simulador de fútbol, con novedades como los equipos Classic XI y mejoras jugables."

    if "liga fantasy" in q or "laliga fantasy" in q:
        return "LALIGA Fantasy es el manager oficial de LALIGA, donde puedes crear tu equipo y competir con amigos."

    if "classic xi" in q or "equipos clasicos" in q:
        return "En FC 26 puedes jugar con los Classic XI, equipos legendarios llenos de estrellas históricas."

    if "eventos especiales" in q or "clasico fantasy" in q:
        return "LALIGA Fantasy organiza eventos especiales como El Clásico, El Derbi de Madrid o El Gran Derbi."

    if "modo carrera" in q or "career mode" in q:
        return "En FC 26 el Modo Carrera te permite gestionar un club o vivir la carrera de un jugador."

    if "ultimate team" in q or "fut" in q:
        return "Ultimate Team en FC 26 sigue siendo el modo estrella para crear tu plantilla soñada."

    if "volta" in q or "futbol callejero" in q:
        return "VOLTA en FC 26 trae el fútbol callejero con estilo y jugadas espectaculares."

    if "clasico" in q and "fantasy" in q:
        return "En LALIGA Fantasy puedes vivir El Clásico con puntuaciones especiales y retos únicos."

    if "derbi" in q and "fantasy" in q:
        return "Los derbis en LALIGA Fantasy son emocionantes, con premios y puntuaciones extra."

    if "capitan fantasy" in q or "doble puntuacion" in q:
        return "En LALIGA Fantasy tu capitán puntúa doble, eligiendo bien puedes ganar la jornada."

    if "banquillo fantasy" in q or "alineacion fantasy" in q:
        return "En LALIGA Fantasy puedes usar tu banquillo y ajustar la alineación para maximizar puntos."

    if "fichajes fantasy" in q or "mercado fantasy" in q:
        return "El mercado de LALIGA Fantasy te permite fichar y vender jugadores según su rendimiento real."

    if "gilberto mora" in q:
        return "Gilberto Mora debutó en FC 26 como una joven promesa con gran potencial."

    if "estadisticas fantasy" in q or "puntos fantasy" in q:
        return "Las estadísticas de LALIGA Fantasy se basan en el rendimiento real de los jugadores cada jornada."

    if "portada fc 26" in q or "cover fc 26" in q:
        return "La portada de FC 26 destaca a jóvenes estrellas como Bellingham y Musiala."

    if "ventas fc 26" in q or "exito fc 26" in q:
        return "FC 26 arrasó en ventas físicas en España, liderando el mercado en PS5."

    if "jugabilidad fc 26" in q or "gameplay fc 26" in q:
        return "La jugabilidad de FC 26 se refinó gracias a los comentarios de la comunidad."

    if "promesas fc 26" in q or "jugadores jovenes fc 26" in q:
        return "En FC 26 aparecen jóvenes promesas como Gilberto Mora, con gran potencial de crecimiento."

    if "liga fantasy premios" in q or "recompensas fantasy" in q:
        return "En LALIGA Fantasy se reparten premios cada jornada según tu rendimiento."

    if "liga fantasy premium" in q or "fantasy premium" in q:
        return "La versión premium de LALIGA Fantasy incluye capitán con doble puntuación, banquillo y entrenador."

    if "liga fantasy clasico" in q or "evento clasico fantasy" in q:
        return "En LALIGA Fantasy puedes vivir El Clásico con puntuaciones y retos especiales."

    if "liga fantasy derbi" in q or "evento derbi fantasy" in q:
        return "Los derbis en LALIGA Fantasy son emocionantes, con bonificaciones y desafíos únicos."

    if "liga fantasy fichajes" in q or "mercado fantasy" in q:
        return "El mercado de LALIGA Fantasy te permite fichar y vender jugadores según su rendimiento real."

    if "liga fantasy temporada" in q or "fantasy 25/26" in q:
        return "La temporada 2025/26 de LALIGA Fantasy incluye fichajes actualizados y nuevas estrellas como Mbappé y Lamine Yamal."

    if "segunda division" in q or "laliga hypermotion" in q:
        return "La Segunda División española, ahora llamada LaLiga Hypermotion, es donde los equipos luchan por subir a Primera."

    if "ascenso" in q or "subir a primera" in q:
        return "El ascenso en LaLiga Hypermotion es el sueño de todos los equipos, con playoffs llenos de emoción."

    if "descenso" in q or "bajar a segunda" in q:
        return "El descenso siempre es duro, pero forma parte de la emoción de las ligas españolas."

    if "playoffs segunda" in q or "promocion segunda" in q:
        return "Los playoffs de ascenso en Segunda son partidos de máxima tensión y emoción."

    if "liga femenina" in q or "liga f" in q:
        return "La Liga F es la máxima categoría del fútbol femenino en España, llena de talento y pasión."

    if "seleccion femenina" in q or "espana femenina" in q:
        return "La selección femenina de España es campeona del mundo, un orgullo para el fútbol español."

    if "champions femenina" in q or "uwcl" in q:
        return "La Champions femenina es el torneo más prestigioso de clubes, donde el Barça ha brillado en los últimos años, aunque el equipo con más champions es el OL Lyonnes."

    if "equipos historicos segunda" in q or "clasicos segunda" in q:
        return "En Segunda han jugado equipos históricos como Zaragoza, Sporting o Deportivo, con gran tradición."

    if "partidos de segunda" in q or "jornada segunda" in q:
        return "Cada jornada de LaLiga Hypermotion es clave, porque todos buscan subir o evitar el descenso."

    if "futsal" in q or "futbol sala" in q:
        return "El futsal es fútbol en espacio reducido, lleno de técnica y velocidad."

    if "liga nacional de futbol sala" in q or "lnfs" in q:
        return "La LNFS es la liga más importante de futsal en España, con equipos históricos como Inter Movistar y Barça."

    if "mundial futsal" in q or "copa del mundo futsal" in q:
        return "El Mundial de futsal reúne a las mejores selecciones del mundo en un espectáculo único."

    if "seleccion española futsal" in q or "espana futsal" in q:
        return "La selección española de futsal es una potencia mundial, con múltiples títulos europeos y mundiales."

    if "inter movistar" in q or "movistar inter" in q:
        return "Movistar Inter es uno de los clubes más exitosos del futsal, con muchos títulos nacionales e internacionales."

    if "barça futsal" in q or "barcelona futsal" in q:
        return "El Barça futsal es un referente en España y Europa, con gran talento en su plantilla."

    if "ricardinho" in q:
        return "Ricardinho es considerado uno de los mejores jugadores de futsal de la historia, con magia en cada jugada."

    if "partido futsal" in q or "jornada futsal" in q:
        return "Los partidos de futsal son rápidos y emocionantes, cada jugada puede terminar en gol."

    if "tecnica futsal" in q or "habilidad futsal" in q:
        return "El futsal destaca por la técnica individual, el control del balón y las jugadas espectaculares."

    if "champions futsal" in q or "uefa futsal" in q:
        return "La UEFA Futsal Champions League es el torneo más prestigioso de clubes en Europa."

    if "wwe" in q or "lucha libre" in q:
        return "Esto no es WWE, pero en el fútbol también hay choques que parecen combates."

    if "john cena" in q or "the rock" in q:
        return "John Cena y The Rock son estrellas de WWE, pero en el fútbol los ídolos también levantan pasiones."

    if "undertaker" in q:
        return "El Undertaker dominaba el ring, igual que algunos equipos dominan el campo de fútbol."

    if "naruto" in q:
        return "Naruto soñaba con ser Hokage, igual que muchos sueñan con ser campeones de liga."

    if "sasuke" in q:
        return "Sasuke buscaba poder, como un delantero que siempre quiere marcar más goles."

    if "kamehameha" in q or "rasengan" in q:
        return "Eso suena más a anime, pero en el fútbol también hay tiros que parecen poderes especiales."

    if "uchiha" in q or "sharingan" in q:
        return "El Sharingan todo lo ve, como un buen mediocentro que controla el partido."

    if "wrestlemania" in q:
        return "WrestleMania es el gran evento de WWE, como una final de Champions en el fútbol."

    if "anime" in q or "manga" in q:
        return "El anime tiene batallas épicas, igual que el fútbol tiene partidos inolvidables."

    if "hokage" in q:
        return "Ser Hokage en Naruto es como levantar la Copa del Mundo en fútbol: el máximo sueño."

    if "triple h" in q or "pedigree" in q:
        return "El 'Pedigree' de Triple H es letal en WWE, como un golazo en el último minuto."

    if "rey mysterio" in q or "619" in q:
        return "El 619 de Rey Mysterio es pura agilidad, igual que un regate eléctrico en fútbol."

    if "roman reigns" in q or "jefe tribal" in q:
        return "Roman Reigns domina WWE como un capitán que manda en el vestuario de fútbol."

    if "naruto vs sasuke" in q:
        return "Naruto vs Sasuke es como un Clásico Barça-Madrid: rivalidad eterna y llena de emoción."

    if "itachi" in q or "uchiha" in q:
        return "Itachi veía todo con el Sharingan, como un mediocentro que controla el ritmo del partido."

    if "madara" in q:
        return "Madara Uchiha era imparable, como un delantero que no deja de marcar goles."

    if "jiraiya" in q or "sabio" in q:
        return "Jiraiya enseñaba a Naruto, igual que un buen entrenador guía a su equipo."

    if "wrestlemania" in q:
        return "WrestleMania es el evento máximo de WWE, como una final de Champions en fútbol."

    if "hokage" in q:
        return "Ser Hokage en Naruto es como levantar la Copa del Mundo: el sueño más grande."

    if "jutsu" in q or "tecnica ninja" in q:
        return "Los jutsus en Naruto son como las jugadas ensayadas en fútbol: pura estrategia y sorpresa."

    if "brock lesnar" in q or "suplex" in q:
        return "Brock Lesnar hacía suplex en WWE, como un defensa que despeja con fuerza cada balón."

    if "randy orton" in q or "rko" in q:
        return "El RKO de Randy Orton es inesperado, como un gol de chilena en el último minuto."

    if "kane" in q or "demonio rojo" in q:
        return "Kane imponía respeto en WWE, igual que un portero que intimida a los delanteros."

    if "naruto run" in q or "correr como naruto" in q:
        return "Correr como Naruto es como un extremo desbordando por la banda con velocidad imparable."

    if "gaara" in q or "arena" in q:
        return "Gaara controlaba la arena, como un mediocentro que controla el ritmo del partido."

    if "rock lee" in q or "taijutsu" in q:
        return "Rock Lee entrenaba sin descanso, como un jugador que nunca se rinde en el campo."

    if "orochimaru" in q or "serpiente" in q:
        return "Orochimaru era astuto y peligroso, como un delantero que aparece donde menos lo esperas."

    if "naruto shippuden" in q or "shippuden" in q:
        return "Naruto Shippuden mostró batallas épicas, como las finales de Champions en fútbol."

    if "campeon wwe" in q or "titulo wwe" in q:
        return "Ser campeón en WWE es como levantar la Copa del Mundo en fútbol: gloria absoluta."

    if "akatsuki" in q or "villanos naruto" in q:
        return "La Akatsuki era temida en Naruto, como un equipo rival que nadie quiere enfrentar."

    if "haku" in q:
        return "Haku dominaba el hielo en Naruto, como un portero que congela cada intento de gol."

    if "stephanie vaquer" in q:
        return "Stephanie Vaquer, 'La Primera', como la hinchada que abre el camino y nunca deja de alentar."

    if "rhea ripley" in q:
        return "Rhea Ripley juega con Brutalidad, como un mediocentro que barre todo lo que pasa por su zona."

    if "dominik mysterio" in q:
        return "El sucio Dom, el ginecólogo del ring, es como ese equipo que siempre hace trampas para ganar."

    if "tenten" in q:
        return "Tenten dominaba las armas ninja, como un jugador que domina todas las posiciones en el campo."

    if "bron breakker" in q:
        return "Bron Breakker es pura fuerza en WWE, como un delantero tanque que arrasa defensas."

    if "jey uso" in q or "uso" in q:
        return "Four letters, one word, YEET!"

    if "chelsea green" in q:
        return "Chelsea Green destaca en WWE, como una jugadora que siempre sorprende con su estilo."

    if "kiba" in q or "akamaru" in q:
        return "Kiba y Akamaru eran inseparables, como un dúo de delanteros que siempre juegan en pareja."

    if "sheamus" in q or "brogue kick" in q:
        return "El Brogue Kick de Sheamus es devastador, como un disparo de fuera del área que rompe la red."

    if "itachi sacrificio" in q or "itachi hermano" in q:
        return "El sacrificio de Itachi por Sasuke es como un capitán que da todo por su equipo."

    if "itachi genjutsu" in q or "genjutsu" in q:
        return "El genjutsu de Itachi confundía rivales, como una jugada táctica que descoloca a la defensa."

    return None


def get_top_entity(question: str) -> str | None:
    q = question.lower()
    for keyword, (table, column, label) in STAT_MAP.items():
        if keyword in q:
            rows = supabase.table(table).select("player_id, " + column).order(column, desc=True).limit(1).execute().data
            if not rows:
                return f"No se encuentra información sobre {label}."
            top = rows[0]
            player_row = supabase.table("players").select("name").eq("id", top["player_id"]).execute().data
            player_name = player_row[0]["name"] if player_row else "desconocido"
            return f"El jugador con más {label} es {player_name}, con {top[column]} {label}."
    return None

def get_competition_info(question: str) -> str | None:
    q = question.lower()
    competitions = supabase.table("competitions").select("id, name, season, type, gender, active").execute().data
    for comp in competitions:
        if comp["name"].lower() in q:
            return (f"La competición {comp['name']} (temporada {comp['season']}), "
                    f"tipo {comp['type']}, género {comp['gender']}, activa: {comp['active']}.")
    return None

def get_information_team_info(question: str) -> str | None:
    q = question.lower()
    teams = supabase.table("information_team").select("id, name, city, province, president, founded_year, stadium").execute().data
    for team in teams:
        if team["name"].lower() in q:
            return (f"El equipo {team['name']} está en {team['city']} ({team['province']}), "
                    f"presidente {team['president']}, fundado en {team['founded_year']}, estadio {team['stadium']}.")
    return None

def get_player_info(question: str) -> str | None:
    q = question.lower()
    players = supabase.table("players").select("id, name, nationality, position, jersey_number, height, weight, team_id").execute().data
    for player in players:
        if player["name"].lower() in q:
            team = supabase.table("teams").select("name").eq("id", player["team_id"]).execute().data
            team_name = team[0]["name"] if team else "desconocido"
            return (f"{player['name']} juega como {player['position']}, dorsal {player['jersey_number']}, "
                    f"nacionalidad {player['nationality']}, altura {player['height']}m, peso {player['weight']}kg, "
                    f"equipo {team_name}.")
    return None

def get_referee_info(question: str) -> str | None:
    q = question.lower()
    referees = supabase.table("referees").select("id, name, nationality, debut").execute().data
    for ref in referees:
        if ref["name"].lower() in q:
            return (f"Árbitro {ref['name']}, nacionalidad {ref['nationality']}, debut {ref['debut']}.")
    return None

def get_referee_stats_info(question: str) -> str | None:
    q = question.lower()
    stats = supabase.table("referee_stats").select("referee_id, yellow_cards, red_cards, wins, draws, defeats").execute().data
    for stat in stats:
        ref = supabase.table("referees").select("name").eq("id", stat["referee_id"]).execute().data
        ref_name = ref[0]["name"] if ref else "desconocido"
        if ref_name.lower() in q:
            return (f"Estadísticas de {ref_name}: amarillas {stat['yellow_cards']}, rojas {stat['red_cards']}, "
                    f"victorias {stat['wins']}, empates {stat['draws']}, derrotas {stat['defeats']}.")
    return None

def get_stadium_info(question: str) -> str | None:
    q = question.lower()
    stadiums = supabase.table("stadiums").select("id, name, city, capacity, year_construction").execute().data
    for stadium in stadiums:
        if stadium["name"].lower() in q:
            return (f"El estadio {stadium['name']} está en {stadium['city']}, "
                    f"capacidad {stadium['capacity']}, construido en {stadium['year_construction']}.")
    return None

def get_team_by_stadium(question: str) -> str | None:
    q = normalize(question)
    stadiums = supabase.table("stadiums").select("id, name").execute().data
    for stadium in stadiums:
        if normalize(stadium["name"]) in q:
            team = supabase.table("teams").select("name").eq("stadium_id", stadium["id"]).execute().data
            if team:
                return f"El equipo que juega en {stadium['name']} es {team[0]['name']}."
            else:
                return f"No se encuentra equipo asociado al estadio {stadium['name']}."
    return None


def get_player_stats(question: str) -> str | None:
    q = question.lower()
    stats = supabase.table("stats").select("player_id, goals, assists, games_played, yellow_card, red_card").execute().data
    for stat in stats:
        player = supabase.table("players").select("name").eq("id", stat["player_id"]).execute().data
        player_name = player[0]["name"] if player else "desconocido"
        if player_name.lower() in q:
            return (f"Estadísticas de {player_name}: goles {stat['goals']}, asistencias {stat['assists']}, "
                    f"partidos {stat['games_played']}, amarillas {stat['yellow_card']}, rojas {stat['red_card']}.")
    return None

def get_team_info(question: str) -> str | None:
    q = normalize(question)
    teams = supabase.table("teams").select("id, name, city, province, founded_year, stadium_id").execute().data
    for team in teams:
        if normalize(team["name"]) in q:
            stadium = supabase.table("stadiums").select("name").eq("id", team["stadium_id"]).execute().data
            stadium_name = stadium[0]["name"] if stadium else "desconocido"
            return (f"Equipo {team['name']} de {team['city']} ({team['province']}), "
                    f"fundado en {team['founded_year']}, estadio {stadium_name}.")
    return None

def get_team_city(question: str) -> str | None:
    q = normalize(question)
    teams = supabase.table("teams").select("name, city").execute().data
    for team in teams:
        if normalize(team["name"]) in q and "ciudad" in q:
            return f"El {team['name']} está en la ciudad de {team['city']}."
    return None



class ChatRequest(BaseModel):
    question: str

class SourceRef(BaseModel):
    table: str
    id: Any
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    q = normalize(req.question)

    # 1. Máximos/mínimos
    answer = get_top_entity(q)
    if answer:
        return ChatResponse(answer=answer, sources=[])

    # 2. Conversación básica (smalltalk)
    answer = get_smalltalk(q)
    if answer:
        return ChatResponse(answer=answer, sources=[])

    # 3. Funciones específicas
    for func in [
        get_competition_info,
        get_information_team_info,
        get_player_info,
        get_referee_info,
        get_referee_stats_info,
        get_stadium_info,
        get_team_by_stadium,
        get_player_stats,
        get_team_info,
        get_team_city
    ]:
        answer = func(q)
        if answer:
            return ChatResponse(answer=answer, sources=[])

    # 4. Flujo RAG con embeddings
    chunks = retrieve_context(req.question)
    if chunks:
        answer = generate_answer(req.question, chunks)
        sources = [
            SourceRef(
                table=c.get("source_table", "unknown"),
                id=c.get("source_id", "unknown"),
                score=round(c["_score"], 4)
            )
            for c in chunks
        ]
        return ChatResponse(answer=answer, sources=sources)

    # 5. Sin acceso a Internet → solo respuesta local
    return ChatResponse(answer="No hay datos suficientes en la BBDD.", sources=[])



# ===== Ejecución local =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)
