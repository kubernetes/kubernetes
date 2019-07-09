package gherkin

// Builtin dialects for af (Afrikaans), am (Armenian), ar (Arabic), bg (Bulgarian), bm (Malay), bs (Bosnian), ca (Catalan), cs (Czech), cy-GB (Welsh), da (Danish), de (German), el (Greek), em (Emoji), en (English), en-Scouse (Scouse), en-au (Australian), en-lol (LOLCAT), en-old (Old English), en-pirate (Pirate), eo (Esperanto), es (Spanish), et (Estonian), fa (Persian), fi (Finnish), fr (French), ga (Irish), gj (Gujarati), gl (Galician), he (Hebrew), hi (Hindi), hr (Croatian), ht (Creole), hu (Hungarian), id (Indonesian), is (Icelandic), it (Italian), ja (Japanese), jv (Javanese), kn (Kannada), ko (Korean), lt (Lithuanian), lu (Luxemburgish), lv (Latvian), mn (Mongolian), nl (Dutch), no (Norwegian), pa (Panjabi), pl (Polish), pt (Portuguese), ro (Romanian), ru (Russian), sk (Slovak), sl (Slovenian), sr-Cyrl (Serbian), sr-Latn (Serbian (Latin)), sv (Swedish), ta (Tamil), th (Thai), tl (Telugu), tlh (Klingon), tr (Turkish), tt (Tatar), uk (Ukrainian), ur (Urdu), uz (Uzbek), vi (Vietnamese), zh-CN (Chinese simplified), zh-TW (Chinese traditional)
func GherkinDialectsBuildin() GherkinDialectProvider {
	return buildinDialects
}

const (
	feature         = "feature"
	background      = "background"
	scenario        = "scenario"
	scenarioOutline = "scenarioOutline"
	examples        = "examples"
	given           = "given"
	when            = "when"
	then            = "then"
	and             = "and"
	but             = "but"
)

var buildinDialects = gherkinDialectMap{
	"af": &GherkinDialect{
		"af", "Afrikaans", "Afrikaans", map[string][]string{
			and: []string{
				"* ",
				"En ",
			},
			background: []string{
				"Agtergrond",
			},
			but: []string{
				"* ",
				"Maar ",
			},
			examples: []string{
				"Voorbeelde",
			},
			feature: []string{
				"Funksie",
				"Besigheid Behoefte",
				"Vermoë",
			},
			given: []string{
				"* ",
				"Gegewe ",
			},
			scenario: []string{
				"Situasie",
			},
			scenarioOutline: []string{
				"Situasie Uiteensetting",
			},
			then: []string{
				"* ",
				"Dan ",
			},
			when: []string{
				"* ",
				"Wanneer ",
			},
		},
	},
	"am": &GherkinDialect{
		"am", "Armenian", "հայերեն", map[string][]string{
			and: []string{
				"* ",
				"Եվ ",
			},
			background: []string{
				"Կոնտեքստ",
			},
			but: []string{
				"* ",
				"Բայց ",
			},
			examples: []string{
				"Օրինակներ",
			},
			feature: []string{
				"Ֆունկցիոնալություն",
				"Հատկություն",
			},
			given: []string{
				"* ",
				"Դիցուք ",
			},
			scenario: []string{
				"Սցենար",
			},
			scenarioOutline: []string{
				"Սցենարի կառուցվացքը",
			},
			then: []string{
				"* ",
				"Ապա ",
			},
			when: []string{
				"* ",
				"Եթե ",
				"Երբ ",
			},
		},
	},
	"ar": &GherkinDialect{
		"ar", "Arabic", "العربية", map[string][]string{
			and: []string{
				"* ",
				"و ",
			},
			background: []string{
				"الخلفية",
			},
			but: []string{
				"* ",
				"لكن ",
			},
			examples: []string{
				"امثلة",
			},
			feature: []string{
				"خاصية",
			},
			given: []string{
				"* ",
				"بفرض ",
			},
			scenario: []string{
				"سيناريو",
			},
			scenarioOutline: []string{
				"سيناريو مخطط",
			},
			then: []string{
				"* ",
				"اذاً ",
				"ثم ",
			},
			when: []string{
				"* ",
				"متى ",
				"عندما ",
			},
		},
	},
	"bg": &GherkinDialect{
		"bg", "Bulgarian", "български", map[string][]string{
			and: []string{
				"* ",
				"И ",
			},
			background: []string{
				"Предистория",
			},
			but: []string{
				"* ",
				"Но ",
			},
			examples: []string{
				"Примери",
			},
			feature: []string{
				"Функционалност",
			},
			given: []string{
				"* ",
				"Дадено ",
			},
			scenario: []string{
				"Сценарий",
			},
			scenarioOutline: []string{
				"Рамка на сценарий",
			},
			then: []string{
				"* ",
				"То ",
			},
			when: []string{
				"* ",
				"Когато ",
			},
		},
	},
	"bm": &GherkinDialect{
		"bm", "Malay", "Bahasa Melayu", map[string][]string{
			and: []string{
				"* ",
				"Dan ",
			},
			background: []string{
				"Latar Belakang",
			},
			but: []string{
				"* ",
				"Tetapi ",
				"Tapi ",
			},
			examples: []string{
				"Contoh",
			},
			feature: []string{
				"Fungsi",
			},
			given: []string{
				"* ",
				"Diberi ",
				"Bagi ",
			},
			scenario: []string{
				"Senario",
				"Situasi",
				"Keadaan",
			},
			scenarioOutline: []string{
				"Kerangka Senario",
				"Kerangka Situasi",
				"Kerangka Keadaan",
				"Garis Panduan Senario",
			},
			then: []string{
				"* ",
				"Maka ",
				"Kemudian ",
			},
			when: []string{
				"* ",
				"Apabila ",
			},
		},
	},
	"bs": &GherkinDialect{
		"bs", "Bosnian", "Bosanski", map[string][]string{
			and: []string{
				"* ",
				"I ",
				"A ",
			},
			background: []string{
				"Pozadina",
			},
			but: []string{
				"* ",
				"Ali ",
			},
			examples: []string{
				"Primjeri",
			},
			feature: []string{
				"Karakteristika",
			},
			given: []string{
				"* ",
				"Dato ",
			},
			scenario: []string{
				"Scenariju",
				"Scenario",
			},
			scenarioOutline: []string{
				"Scenariju-obris",
				"Scenario-outline",
			},
			then: []string{
				"* ",
				"Zatim ",
			},
			when: []string{
				"* ",
				"Kada ",
			},
		},
	},
	"ca": &GherkinDialect{
		"ca", "Catalan", "català", map[string][]string{
			and: []string{
				"* ",
				"I ",
			},
			background: []string{
				"Rerefons",
				"Antecedents",
			},
			but: []string{
				"* ",
				"Però ",
			},
			examples: []string{
				"Exemples",
			},
			feature: []string{
				"Característica",
				"Funcionalitat",
			},
			given: []string{
				"* ",
				"Donat ",
				"Donada ",
				"Atès ",
				"Atesa ",
			},
			scenario: []string{
				"Escenari",
			},
			scenarioOutline: []string{
				"Esquema de l'escenari",
			},
			then: []string{
				"* ",
				"Aleshores ",
				"Cal ",
			},
			when: []string{
				"* ",
				"Quan ",
			},
		},
	},
	"cs": &GherkinDialect{
		"cs", "Czech", "Česky", map[string][]string{
			and: []string{
				"* ",
				"A také ",
				"A ",
			},
			background: []string{
				"Pozadí",
				"Kontext",
			},
			but: []string{
				"* ",
				"Ale ",
			},
			examples: []string{
				"Příklady",
			},
			feature: []string{
				"Požadavek",
			},
			given: []string{
				"* ",
				"Pokud ",
				"Za předpokladu ",
			},
			scenario: []string{
				"Scénář",
			},
			scenarioOutline: []string{
				"Náčrt Scénáře",
				"Osnova scénáře",
			},
			then: []string{
				"* ",
				"Pak ",
			},
			when: []string{
				"* ",
				"Když ",
			},
		},
	},
	"cy-GB": &GherkinDialect{
		"cy-GB", "Welsh", "Cymraeg", map[string][]string{
			and: []string{
				"* ",
				"A ",
			},
			background: []string{
				"Cefndir",
			},
			but: []string{
				"* ",
				"Ond ",
			},
			examples: []string{
				"Enghreifftiau",
			},
			feature: []string{
				"Arwedd",
			},
			given: []string{
				"* ",
				"Anrhegedig a ",
			},
			scenario: []string{
				"Scenario",
			},
			scenarioOutline: []string{
				"Scenario Amlinellol",
			},
			then: []string{
				"* ",
				"Yna ",
			},
			when: []string{
				"* ",
				"Pryd ",
			},
		},
	},
	"da": &GherkinDialect{
		"da", "Danish", "dansk", map[string][]string{
			and: []string{
				"* ",
				"Og ",
			},
			background: []string{
				"Baggrund",
			},
			but: []string{
				"* ",
				"Men ",
			},
			examples: []string{
				"Eksempler",
			},
			feature: []string{
				"Egenskab",
			},
			given: []string{
				"* ",
				"Givet ",
			},
			scenario: []string{
				"Scenarie",
			},
			scenarioOutline: []string{
				"Abstrakt Scenario",
			},
			then: []string{
				"* ",
				"Så ",
			},
			when: []string{
				"* ",
				"Når ",
			},
		},
	},
	"de": &GherkinDialect{
		"de", "German", "Deutsch", map[string][]string{
			and: []string{
				"* ",
				"Und ",
			},
			background: []string{
				"Grundlage",
			},
			but: []string{
				"* ",
				"Aber ",
			},
			examples: []string{
				"Beispiele",
			},
			feature: []string{
				"Funktionalität",
			},
			given: []string{
				"* ",
				"Angenommen ",
				"Gegeben sei ",
				"Gegeben seien ",
			},
			scenario: []string{
				"Szenario",
			},
			scenarioOutline: []string{
				"Szenariogrundriss",
			},
			then: []string{
				"* ",
				"Dann ",
			},
			when: []string{
				"* ",
				"Wenn ",
			},
		},
	},
	"el": &GherkinDialect{
		"el", "Greek", "Ελληνικά", map[string][]string{
			and: []string{
				"* ",
				"Και ",
			},
			background: []string{
				"Υπόβαθρο",
			},
			but: []string{
				"* ",
				"Αλλά ",
			},
			examples: []string{
				"Παραδείγματα",
				"Σενάρια",
			},
			feature: []string{
				"Δυνατότητα",
				"Λειτουργία",
			},
			given: []string{
				"* ",
				"Δεδομένου ",
			},
			scenario: []string{
				"Σενάριο",
			},
			scenarioOutline: []string{
				"Περιγραφή Σεναρίου",
			},
			then: []string{
				"* ",
				"Τότε ",
			},
			when: []string{
				"* ",
				"Όταν ",
			},
		},
	},
	"em": &GherkinDialect{
		"em", "Emoji", "😀", map[string][]string{
			and: []string{
				"* ",
				"😂",
			},
			background: []string{
				"💤",
			},
			but: []string{
				"* ",
				"😔",
			},
			examples: []string{
				"📓",
			},
			feature: []string{
				"📚",
			},
			given: []string{
				"* ",
				"😐",
			},
			scenario: []string{
				"📕",
			},
			scenarioOutline: []string{
				"📖",
			},
			then: []string{
				"* ",
				"🙏",
			},
			when: []string{
				"* ",
				"🎬",
			},
		},
	},
	"en": &GherkinDialect{
		"en", "English", "English", map[string][]string{
			and: []string{
				"* ",
				"And ",
			},
			background: []string{
				"Background",
			},
			but: []string{
				"* ",
				"But ",
			},
			examples: []string{
				"Examples",
				"Scenarios",
			},
			feature: []string{
				"Feature",
				"Business Need",
				"Ability",
			},
			given: []string{
				"* ",
				"Given ",
			},
			scenario: []string{
				"Scenario",
			},
			scenarioOutline: []string{
				"Scenario Outline",
				"Scenario Template",
			},
			then: []string{
				"* ",
				"Then ",
			},
			when: []string{
				"* ",
				"When ",
			},
		},
	},
	"en-Scouse": &GherkinDialect{
		"en-Scouse", "Scouse", "Scouse", map[string][]string{
			and: []string{
				"* ",
				"An ",
			},
			background: []string{
				"Dis is what went down",
			},
			but: []string{
				"* ",
				"Buh ",
			},
			examples: []string{
				"Examples",
			},
			feature: []string{
				"Feature",
			},
			given: []string{
				"* ",
				"Givun ",
				"Youse know when youse got ",
			},
			scenario: []string{
				"The thing of it is",
			},
			scenarioOutline: []string{
				"Wharrimean is",
			},
			then: []string{
				"* ",
				"Dun ",
				"Den youse gotta ",
			},
			when: []string{
				"* ",
				"Wun ",
				"Youse know like when ",
			},
		},
	},
	"en-au": &GherkinDialect{
		"en-au", "Australian", "Australian", map[string][]string{
			and: []string{
				"* ",
				"Too right ",
			},
			background: []string{
				"First off",
			},
			but: []string{
				"* ",
				"Yeah nah ",
			},
			examples: []string{
				"You'll wanna",
			},
			feature: []string{
				"Pretty much",
			},
			given: []string{
				"* ",
				"Y'know ",
			},
			scenario: []string{
				"Awww, look mate",
			},
			scenarioOutline: []string{
				"Reckon it's like",
			},
			then: []string{
				"* ",
				"But at the end of the day I reckon ",
			},
			when: []string{
				"* ",
				"It's just unbelievable ",
			},
		},
	},
	"en-lol": &GherkinDialect{
		"en-lol", "LOLCAT", "LOLCAT", map[string][]string{
			and: []string{
				"* ",
				"AN ",
			},
			background: []string{
				"B4",
			},
			but: []string{
				"* ",
				"BUT ",
			},
			examples: []string{
				"EXAMPLZ",
			},
			feature: []string{
				"OH HAI",
			},
			given: []string{
				"* ",
				"I CAN HAZ ",
			},
			scenario: []string{
				"MISHUN",
			},
			scenarioOutline: []string{
				"MISHUN SRSLY",
			},
			then: []string{
				"* ",
				"DEN ",
			},
			when: []string{
				"* ",
				"WEN ",
			},
		},
	},
	"en-old": &GherkinDialect{
		"en-old", "Old English", "Englisc", map[string][]string{
			and: []string{
				"* ",
				"Ond ",
				"7 ",
			},
			background: []string{
				"Aer",
				"Ær",
			},
			but: []string{
				"* ",
				"Ac ",
			},
			examples: []string{
				"Se the",
				"Se þe",
				"Se ðe",
			},
			feature: []string{
				"Hwaet",
				"Hwæt",
			},
			given: []string{
				"* ",
				"Thurh ",
				"Þurh ",
				"Ðurh ",
			},
			scenario: []string{
				"Swa",
			},
			scenarioOutline: []string{
				"Swa hwaer swa",
				"Swa hwær swa",
			},
			then: []string{
				"* ",
				"Tha ",
				"Þa ",
				"Ða ",
				"Tha the ",
				"Þa þe ",
				"Ða ðe ",
			},
			when: []string{
				"* ",
				"Tha ",
				"Þa ",
				"Ða ",
			},
		},
	},
	"en-pirate": &GherkinDialect{
		"en-pirate", "Pirate", "Pirate", map[string][]string{
			and: []string{
				"* ",
				"Aye ",
			},
			background: []string{
				"Yo-ho-ho",
			},
			but: []string{
				"* ",
				"Avast! ",
			},
			examples: []string{
				"Dead men tell no tales",
			},
			feature: []string{
				"Ahoy matey!",
			},
			given: []string{
				"* ",
				"Gangway! ",
			},
			scenario: []string{
				"Heave to",
			},
			scenarioOutline: []string{
				"Shiver me timbers",
			},
			then: []string{
				"* ",
				"Let go and haul ",
			},
			when: []string{
				"* ",
				"Blimey! ",
			},
		},
	},
	"eo": &GherkinDialect{
		"eo", "Esperanto", "Esperanto", map[string][]string{
			and: []string{
				"* ",
				"Kaj ",
			},
			background: []string{
				"Fono",
			},
			but: []string{
				"* ",
				"Sed ",
			},
			examples: []string{
				"Ekzemploj",
			},
			feature: []string{
				"Trajto",
			},
			given: []string{
				"* ",
				"Donitaĵo ",
				"Komence ",
			},
			scenario: []string{
				"Scenaro",
				"Kazo",
			},
			scenarioOutline: []string{
				"Konturo de la scenaro",
				"Skizo",
				"Kazo-skizo",
			},
			then: []string{
				"* ",
				"Do ",
			},
			when: []string{
				"* ",
				"Se ",
			},
		},
	},
	"es": &GherkinDialect{
		"es", "Spanish", "español", map[string][]string{
			and: []string{
				"* ",
				"Y ",
				"E ",
			},
			background: []string{
				"Antecedentes",
			},
			but: []string{
				"* ",
				"Pero ",
			},
			examples: []string{
				"Ejemplos",
			},
			feature: []string{
				"Característica",
			},
			given: []string{
				"* ",
				"Dado ",
				"Dada ",
				"Dados ",
				"Dadas ",
			},
			scenario: []string{
				"Escenario",
			},
			scenarioOutline: []string{
				"Esquema del escenario",
			},
			then: []string{
				"* ",
				"Entonces ",
			},
			when: []string{
				"* ",
				"Cuando ",
			},
		},
	},
	"et": &GherkinDialect{
		"et", "Estonian", "eesti keel", map[string][]string{
			and: []string{
				"* ",
				"Ja ",
			},
			background: []string{
				"Taust",
			},
			but: []string{
				"* ",
				"Kuid ",
			},
			examples: []string{
				"Juhtumid",
			},
			feature: []string{
				"Omadus",
			},
			given: []string{
				"* ",
				"Eeldades ",
			},
			scenario: []string{
				"Stsenaarium",
			},
			scenarioOutline: []string{
				"Raamstsenaarium",
			},
			then: []string{
				"* ",
				"Siis ",
			},
			when: []string{
				"* ",
				"Kui ",
			},
		},
	},
	"fa": &GherkinDialect{
		"fa", "Persian", "فارسی", map[string][]string{
			and: []string{
				"* ",
				"و ",
			},
			background: []string{
				"زمینه",
			},
			but: []string{
				"* ",
				"اما ",
			},
			examples: []string{
				"نمونه ها",
			},
			feature: []string{
				"وِیژگی",
			},
			given: []string{
				"* ",
				"با فرض ",
			},
			scenario: []string{
				"سناریو",
			},
			scenarioOutline: []string{
				"الگوی سناریو",
			},
			then: []string{
				"* ",
				"آنگاه ",
			},
			when: []string{
				"* ",
				"هنگامی ",
			},
		},
	},
	"fi": &GherkinDialect{
		"fi", "Finnish", "suomi", map[string][]string{
			and: []string{
				"* ",
				"Ja ",
			},
			background: []string{
				"Tausta",
			},
			but: []string{
				"* ",
				"Mutta ",
			},
			examples: []string{
				"Tapaukset",
			},
			feature: []string{
				"Ominaisuus",
			},
			given: []string{
				"* ",
				"Oletetaan ",
			},
			scenario: []string{
				"Tapaus",
			},
			scenarioOutline: []string{
				"Tapausaihio",
			},
			then: []string{
				"* ",
				"Niin ",
			},
			when: []string{
				"* ",
				"Kun ",
			},
		},
	},
	"fr": &GherkinDialect{
		"fr", "French", "français", map[string][]string{
			and: []string{
				"* ",
				"Et que ",
				"Et qu'",
				"Et ",
			},
			background: []string{
				"Contexte",
			},
			but: []string{
				"* ",
				"Mais que ",
				"Mais qu'",
				"Mais ",
			},
			examples: []string{
				"Exemples",
			},
			feature: []string{
				"Fonctionnalité",
			},
			given: []string{
				"* ",
				"Soit ",
				"Etant donné que ",
				"Etant donné qu'",
				"Etant donné ",
				"Etant donnée ",
				"Etant donnés ",
				"Etant données ",
				"Étant donné que ",
				"Étant donné qu'",
				"Étant donné ",
				"Étant donnée ",
				"Étant donnés ",
				"Étant données ",
			},
			scenario: []string{
				"Scénario",
			},
			scenarioOutline: []string{
				"Plan du scénario",
				"Plan du Scénario",
			},
			then: []string{
				"* ",
				"Alors ",
			},
			when: []string{
				"* ",
				"Quand ",
				"Lorsque ",
				"Lorsqu'",
			},
		},
	},
	"ga": &GherkinDialect{
		"ga", "Irish", "Gaeilge", map[string][]string{
			and: []string{
				"* ",
				"Agus",
			},
			background: []string{
				"Cúlra",
			},
			but: []string{
				"* ",
				"Ach",
			},
			examples: []string{
				"Samplaí",
			},
			feature: []string{
				"Gné",
			},
			given: []string{
				"* ",
				"Cuir i gcás go",
				"Cuir i gcás nach",
				"Cuir i gcás gur",
				"Cuir i gcás nár",
			},
			scenario: []string{
				"Cás",
			},
			scenarioOutline: []string{
				"Cás Achomair",
			},
			then: []string{
				"* ",
				"Ansin",
			},
			when: []string{
				"* ",
				"Nuair a",
				"Nuair nach",
				"Nuair ba",
				"Nuair nár",
			},
		},
	},
	"gj": &GherkinDialect{
		"gj", "Gujarati", "ગુજરાતી", map[string][]string{
			and: []string{
				"* ",
				"અને ",
			},
			background: []string{
				"બેકગ્રાઉન્ડ",
			},
			but: []string{
				"* ",
				"પણ ",
			},
			examples: []string{
				"ઉદાહરણો",
			},
			feature: []string{
				"લક્ષણ",
				"વ્યાપાર જરૂર",
				"ક્ષમતા",
			},
			given: []string{
				"* ",
				"આપેલ છે ",
			},
			scenario: []string{
				"સ્થિતિ",
			},
			scenarioOutline: []string{
				"પરિદ્દશ્ય રૂપરેખા",
				"પરિદ્દશ્ય ઢાંચો",
			},
			then: []string{
				"* ",
				"પછી ",
			},
			when: []string{
				"* ",
				"ક્યારે ",
			},
		},
	},
	"gl": &GherkinDialect{
		"gl", "Galician", "galego", map[string][]string{
			and: []string{
				"* ",
				"E ",
			},
			background: []string{
				"Contexto",
			},
			but: []string{
				"* ",
				"Mais ",
				"Pero ",
			},
			examples: []string{
				"Exemplos",
			},
			feature: []string{
				"Característica",
			},
			given: []string{
				"* ",
				"Dado ",
				"Dada ",
				"Dados ",
				"Dadas ",
			},
			scenario: []string{
				"Escenario",
			},
			scenarioOutline: []string{
				"Esbozo do escenario",
			},
			then: []string{
				"* ",
				"Entón ",
				"Logo ",
			},
			when: []string{
				"* ",
				"Cando ",
			},
		},
	},
	"he": &GherkinDialect{
		"he", "Hebrew", "עברית", map[string][]string{
			and: []string{
				"* ",
				"וגם ",
			},
			background: []string{
				"רקע",
			},
			but: []string{
				"* ",
				"אבל ",
			},
			examples: []string{
				"דוגמאות",
			},
			feature: []string{
				"תכונה",
			},
			given: []string{
				"* ",
				"בהינתן ",
			},
			scenario: []string{
				"תרחיש",
			},
			scenarioOutline: []string{
				"תבנית תרחיש",
			},
			then: []string{
				"* ",
				"אז ",
				"אזי ",
			},
			when: []string{
				"* ",
				"כאשר ",
			},
		},
	},
	"hi": &GherkinDialect{
		"hi", "Hindi", "हिंदी", map[string][]string{
			and: []string{
				"* ",
				"और ",
				"तथा ",
			},
			background: []string{
				"पृष्ठभूमि",
			},
			but: []string{
				"* ",
				"पर ",
				"परन्तु ",
				"किन्तु ",
			},
			examples: []string{
				"उदाहरण",
			},
			feature: []string{
				"रूप लेख",
			},
			given: []string{
				"* ",
				"अगर ",
				"यदि ",
				"चूंकि ",
			},
			scenario: []string{
				"परिदृश्य",
			},
			scenarioOutline: []string{
				"परिदृश्य रूपरेखा",
			},
			then: []string{
				"* ",
				"तब ",
				"तदा ",
			},
			when: []string{
				"* ",
				"जब ",
				"कदा ",
			},
		},
	},
	"hr": &GherkinDialect{
		"hr", "Croatian", "hrvatski", map[string][]string{
			and: []string{
				"* ",
				"I ",
			},
			background: []string{
				"Pozadina",
			},
			but: []string{
				"* ",
				"Ali ",
			},
			examples: []string{
				"Primjeri",
				"Scenariji",
			},
			feature: []string{
				"Osobina",
				"Mogućnost",
				"Mogucnost",
			},
			given: []string{
				"* ",
				"Zadan ",
				"Zadani ",
				"Zadano ",
			},
			scenario: []string{
				"Scenarij",
			},
			scenarioOutline: []string{
				"Skica",
				"Koncept",
			},
			then: []string{
				"* ",
				"Onda ",
			},
			when: []string{
				"* ",
				"Kada ",
				"Kad ",
			},
		},
	},
	"ht": &GherkinDialect{
		"ht", "Creole", "kreyòl", map[string][]string{
			and: []string{
				"* ",
				"Ak ",
				"Epi ",
				"E ",
			},
			background: []string{
				"Kontèks",
				"Istorik",
			},
			but: []string{
				"* ",
				"Men ",
			},
			examples: []string{
				"Egzanp",
			},
			feature: []string{
				"Karakteristik",
				"Mak",
				"Fonksyonalite",
			},
			given: []string{
				"* ",
				"Sipoze ",
				"Sipoze ke ",
				"Sipoze Ke ",
			},
			scenario: []string{
				"Senaryo",
			},
			scenarioOutline: []string{
				"Plan senaryo",
				"Plan Senaryo",
				"Senaryo deskripsyon",
				"Senaryo Deskripsyon",
				"Dyagram senaryo",
				"Dyagram Senaryo",
			},
			then: []string{
				"* ",
				"Lè sa a ",
				"Le sa a ",
			},
			when: []string{
				"* ",
				"Lè ",
				"Le ",
			},
		},
	},
	"hu": &GherkinDialect{
		"hu", "Hungarian", "magyar", map[string][]string{
			and: []string{
				"* ",
				"És ",
			},
			background: []string{
				"Háttér",
			},
			but: []string{
				"* ",
				"De ",
			},
			examples: []string{
				"Példák",
			},
			feature: []string{
				"Jellemző",
			},
			given: []string{
				"* ",
				"Amennyiben ",
				"Adott ",
			},
			scenario: []string{
				"Forgatókönyv",
			},
			scenarioOutline: []string{
				"Forgatókönyv vázlat",
			},
			then: []string{
				"* ",
				"Akkor ",
			},
			when: []string{
				"* ",
				"Majd ",
				"Ha ",
				"Amikor ",
			},
		},
	},
	"id": &GherkinDialect{
		"id", "Indonesian", "Bahasa Indonesia", map[string][]string{
			and: []string{
				"* ",
				"Dan ",
			},
			background: []string{
				"Dasar",
			},
			but: []string{
				"* ",
				"Tapi ",
			},
			examples: []string{
				"Contoh",
			},
			feature: []string{
				"Fitur",
			},
			given: []string{
				"* ",
				"Dengan ",
			},
			scenario: []string{
				"Skenario",
			},
			scenarioOutline: []string{
				"Skenario konsep",
			},
			then: []string{
				"* ",
				"Maka ",
			},
			when: []string{
				"* ",
				"Ketika ",
			},
		},
	},
	"is": &GherkinDialect{
		"is", "Icelandic", "Íslenska", map[string][]string{
			and: []string{
				"* ",
				"Og ",
			},
			background: []string{
				"Bakgrunnur",
			},
			but: []string{
				"* ",
				"En ",
			},
			examples: []string{
				"Dæmi",
				"Atburðarásir",
			},
			feature: []string{
				"Eiginleiki",
			},
			given: []string{
				"* ",
				"Ef ",
			},
			scenario: []string{
				"Atburðarás",
			},
			scenarioOutline: []string{
				"Lýsing Atburðarásar",
				"Lýsing Dæma",
			},
			then: []string{
				"* ",
				"Þá ",
			},
			when: []string{
				"* ",
				"Þegar ",
			},
		},
	},
	"it": &GherkinDialect{
		"it", "Italian", "italiano", map[string][]string{
			and: []string{
				"* ",
				"E ",
			},
			background: []string{
				"Contesto",
			},
			but: []string{
				"* ",
				"Ma ",
			},
			examples: []string{
				"Esempi",
			},
			feature: []string{
				"Funzionalità",
			},
			given: []string{
				"* ",
				"Dato ",
				"Data ",
				"Dati ",
				"Date ",
			},
			scenario: []string{
				"Scenario",
			},
			scenarioOutline: []string{
				"Schema dello scenario",
			},
			then: []string{
				"* ",
				"Allora ",
			},
			when: []string{
				"* ",
				"Quando ",
			},
		},
	},
	"ja": &GherkinDialect{
		"ja", "Japanese", "日本語", map[string][]string{
			and: []string{
				"* ",
				"かつ",
			},
			background: []string{
				"背景",
			},
			but: []string{
				"* ",
				"しかし",
				"但し",
				"ただし",
			},
			examples: []string{
				"例",
				"サンプル",
			},
			feature: []string{
				"フィーチャ",
				"機能",
			},
			given: []string{
				"* ",
				"前提",
			},
			scenario: []string{
				"シナリオ",
			},
			scenarioOutline: []string{
				"シナリオアウトライン",
				"シナリオテンプレート",
				"テンプレ",
				"シナリオテンプレ",
			},
			then: []string{
				"* ",
				"ならば",
			},
			when: []string{
				"* ",
				"もし",
			},
		},
	},
	"jv": &GherkinDialect{
		"jv", "Javanese", "Basa Jawa", map[string][]string{
			and: []string{
				"* ",
				"Lan ",
			},
			background: []string{
				"Dasar",
			},
			but: []string{
				"* ",
				"Tapi ",
				"Nanging ",
				"Ananging ",
			},
			examples: []string{
				"Conto",
				"Contone",
			},
			feature: []string{
				"Fitur",
			},
			given: []string{
				"* ",
				"Nalika ",
				"Nalikaning ",
			},
			scenario: []string{
				"Skenario",
			},
			scenarioOutline: []string{
				"Konsep skenario",
			},
			then: []string{
				"* ",
				"Njuk ",
				"Banjur ",
			},
			when: []string{
				"* ",
				"Manawa ",
				"Menawa ",
			},
		},
	},
	"kn": &GherkinDialect{
		"kn", "Kannada", "ಕನ್ನಡ", map[string][]string{
			and: []string{
				"* ",
				"ಮತ್ತು ",
			},
			background: []string{
				"ಹಿನ್ನೆಲೆ",
			},
			but: []string{
				"* ",
				"ಆದರೆ ",
			},
			examples: []string{
				"ಉದಾಹರಣೆಗಳು",
			},
			feature: []string{
				"ಹೆಚ್ಚಳ",
			},
			given: []string{
				"* ",
				"ನೀಡಿದ ",
			},
			scenario: []string{
				"ಕಥಾಸಾರಾಂಶ",
			},
			scenarioOutline: []string{
				"ವಿವರಣೆ",
			},
			then: []string{
				"* ",
				"ನಂತರ ",
			},
			when: []string{
				"* ",
				"ಸ್ಥಿತಿಯನ್ನು ",
			},
		},
	},
	"ko": &GherkinDialect{
		"ko", "Korean", "한국어", map[string][]string{
			and: []string{
				"* ",
				"그리고",
			},
			background: []string{
				"배경",
			},
			but: []string{
				"* ",
				"하지만",
				"단",
			},
			examples: []string{
				"예",
			},
			feature: []string{
				"기능",
			},
			given: []string{
				"* ",
				"조건",
				"먼저",
			},
			scenario: []string{
				"시나리오",
			},
			scenarioOutline: []string{
				"시나리오 개요",
			},
			then: []string{
				"* ",
				"그러면",
			},
			when: []string{
				"* ",
				"만일",
				"만약",
			},
		},
	},
	"lt": &GherkinDialect{
		"lt", "Lithuanian", "lietuvių kalba", map[string][]string{
			and: []string{
				"* ",
				"Ir ",
			},
			background: []string{
				"Kontekstas",
			},
			but: []string{
				"* ",
				"Bet ",
			},
			examples: []string{
				"Pavyzdžiai",
				"Scenarijai",
				"Variantai",
			},
			feature: []string{
				"Savybė",
			},
			given: []string{
				"* ",
				"Duota ",
			},
			scenario: []string{
				"Scenarijus",
			},
			scenarioOutline: []string{
				"Scenarijaus šablonas",
			},
			then: []string{
				"* ",
				"Tada ",
			},
			when: []string{
				"* ",
				"Kai ",
			},
		},
	},
	"lu": &GherkinDialect{
		"lu", "Luxemburgish", "Lëtzebuergesch", map[string][]string{
			and: []string{
				"* ",
				"an ",
				"a ",
			},
			background: []string{
				"Hannergrond",
			},
			but: []string{
				"* ",
				"awer ",
				"mä ",
			},
			examples: []string{
				"Beispiller",
			},
			feature: []string{
				"Funktionalitéit",
			},
			given: []string{
				"* ",
				"ugeholl ",
			},
			scenario: []string{
				"Szenario",
			},
			scenarioOutline: []string{
				"Plang vum Szenario",
			},
			then: []string{
				"* ",
				"dann ",
			},
			when: []string{
				"* ",
				"wann ",
			},
		},
	},
	"lv": &GherkinDialect{
		"lv", "Latvian", "latviešu", map[string][]string{
			and: []string{
				"* ",
				"Un ",
			},
			background: []string{
				"Konteksts",
				"Situācija",
			},
			but: []string{
				"* ",
				"Bet ",
			},
			examples: []string{
				"Piemēri",
				"Paraugs",
			},
			feature: []string{
				"Funkcionalitāte",
				"Fīča",
			},
			given: []string{
				"* ",
				"Kad ",
			},
			scenario: []string{
				"Scenārijs",
			},
			scenarioOutline: []string{
				"Scenārijs pēc parauga",
			},
			then: []string{
				"* ",
				"Tad ",
			},
			when: []string{
				"* ",
				"Ja ",
			},
		},
	},
	"mn": &GherkinDialect{
		"mn", "Mongolian", "монгол", map[string][]string{
			and: []string{
				"* ",
				"Мөн ",
				"Тэгээд ",
			},
			background: []string{
				"Агуулга",
			},
			but: []string{
				"* ",
				"Гэхдээ ",
				"Харин ",
			},
			examples: []string{
				"Тухайлбал",
			},
			feature: []string{
				"Функц",
				"Функционал",
			},
			given: []string{
				"* ",
				"Өгөгдсөн нь ",
				"Анх ",
			},
			scenario: []string{
				"Сценар",
			},
			scenarioOutline: []string{
				"Сценарын төлөвлөгөө",
			},
			then: []string{
				"* ",
				"Тэгэхэд ",
				"Үүний дараа ",
			},
			when: []string{
				"* ",
				"Хэрэв ",
			},
		},
	},
	"nl": &GherkinDialect{
		"nl", "Dutch", "Nederlands", map[string][]string{
			and: []string{
				"* ",
				"En ",
			},
			background: []string{
				"Achtergrond",
			},
			but: []string{
				"* ",
				"Maar ",
			},
			examples: []string{
				"Voorbeelden",
			},
			feature: []string{
				"Functionaliteit",
			},
			given: []string{
				"* ",
				"Gegeven ",
				"Stel ",
			},
			scenario: []string{
				"Scenario",
			},
			scenarioOutline: []string{
				"Abstract Scenario",
			},
			then: []string{
				"* ",
				"Dan ",
			},
			when: []string{
				"* ",
				"Als ",
			},
		},
	},
	"no": &GherkinDialect{
		"no", "Norwegian", "norsk", map[string][]string{
			and: []string{
				"* ",
				"Og ",
			},
			background: []string{
				"Bakgrunn",
			},
			but: []string{
				"* ",
				"Men ",
			},
			examples: []string{
				"Eksempler",
			},
			feature: []string{
				"Egenskap",
			},
			given: []string{
				"* ",
				"Gitt ",
			},
			scenario: []string{
				"Scenario",
			},
			scenarioOutline: []string{
				"Scenariomal",
				"Abstrakt Scenario",
			},
			then: []string{
				"* ",
				"Så ",
			},
			when: []string{
				"* ",
				"Når ",
			},
		},
	},
	"pa": &GherkinDialect{
		"pa", "Panjabi", "ਪੰਜਾਬੀ", map[string][]string{
			and: []string{
				"* ",
				"ਅਤੇ ",
			},
			background: []string{
				"ਪਿਛੋਕੜ",
			},
			but: []string{
				"* ",
				"ਪਰ ",
			},
			examples: []string{
				"ਉਦਾਹਰਨਾਂ",
			},
			feature: []string{
				"ਖਾਸੀਅਤ",
				"ਮੁਹਾਂਦਰਾ",
				"ਨਕਸ਼ ਨੁਹਾਰ",
			},
			given: []string{
				"* ",
				"ਜੇਕਰ ",
				"ਜਿਵੇਂ ਕਿ ",
			},
			scenario: []string{
				"ਪਟਕਥਾ",
			},
			scenarioOutline: []string{
				"ਪਟਕਥਾ ਢਾਂਚਾ",
				"ਪਟਕਥਾ ਰੂਪ ਰੇਖਾ",
			},
			then: []string{
				"* ",
				"ਤਦ ",
			},
			when: []string{
				"* ",
				"ਜਦੋਂ ",
			},
		},
	},
	"pl": &GherkinDialect{
		"pl", "Polish", "polski", map[string][]string{
			and: []string{
				"* ",
				"Oraz ",
				"I ",
			},
			background: []string{
				"Założenia",
			},
			but: []string{
				"* ",
				"Ale ",
			},
			examples: []string{
				"Przykłady",
			},
			feature: []string{
				"Właściwość",
				"Funkcja",
				"Aspekt",
				"Potrzeba biznesowa",
			},
			given: []string{
				"* ",
				"Zakładając ",
				"Mając ",
				"Zakładając, że ",
			},
			scenario: []string{
				"Scenariusz",
			},
			scenarioOutline: []string{
				"Szablon scenariusza",
			},
			then: []string{
				"* ",
				"Wtedy ",
			},
			when: []string{
				"* ",
				"Jeżeli ",
				"Jeśli ",
				"Gdy ",
				"Kiedy ",
			},
		},
	},
	"pt": &GherkinDialect{
		"pt", "Portuguese", "português", map[string][]string{
			and: []string{
				"* ",
				"E ",
			},
			background: []string{
				"Contexto",
				"Cenário de Fundo",
				"Cenario de Fundo",
				"Fundo",
			},
			but: []string{
				"* ",
				"Mas ",
			},
			examples: []string{
				"Exemplos",
				"Cenários",
				"Cenarios",
			},
			feature: []string{
				"Funcionalidade",
				"Característica",
				"Caracteristica",
			},
			given: []string{
				"* ",
				"Dado ",
				"Dada ",
				"Dados ",
				"Dadas ",
			},
			scenario: []string{
				"Cenário",
				"Cenario",
			},
			scenarioOutline: []string{
				"Esquema do Cenário",
				"Esquema do Cenario",
				"Delineação do Cenário",
				"Delineacao do Cenario",
			},
			then: []string{
				"* ",
				"Então ",
				"Entao ",
			},
			when: []string{
				"* ",
				"Quando ",
			},
		},
	},
	"ro": &GherkinDialect{
		"ro", "Romanian", "română", map[string][]string{
			and: []string{
				"* ",
				"Si ",
				"Și ",
				"Şi ",
			},
			background: []string{
				"Context",
			},
			but: []string{
				"* ",
				"Dar ",
			},
			examples: []string{
				"Exemple",
			},
			feature: []string{
				"Functionalitate",
				"Funcționalitate",
				"Funcţionalitate",
			},
			given: []string{
				"* ",
				"Date fiind ",
				"Dat fiind ",
				"Dati fiind ",
				"Dați fiind ",
				"Daţi fiind ",
			},
			scenario: []string{
				"Scenariu",
			},
			scenarioOutline: []string{
				"Structura scenariu",
				"Structură scenariu",
			},
			then: []string{
				"* ",
				"Atunci ",
			},
			when: []string{
				"* ",
				"Cand ",
				"Când ",
			},
		},
	},
	"ru": &GherkinDialect{
		"ru", "Russian", "русский", map[string][]string{
			and: []string{
				"* ",
				"И ",
				"К тому же ",
				"Также ",
			},
			background: []string{
				"Предыстория",
				"Контекст",
			},
			but: []string{
				"* ",
				"Но ",
				"А ",
			},
			examples: []string{
				"Примеры",
			},
			feature: []string{
				"Функция",
				"Функционал",
				"Свойство",
			},
			given: []string{
				"* ",
				"Допустим ",
				"Дано ",
				"Пусть ",
			},
			scenario: []string{
				"Сценарий",
			},
			scenarioOutline: []string{
				"Структура сценария",
			},
			then: []string{
				"* ",
				"То ",
				"Тогда ",
			},
			when: []string{
				"* ",
				"Если ",
				"Когда ",
			},
		},
	},
	"sk": &GherkinDialect{
		"sk", "Slovak", "Slovensky", map[string][]string{
			and: []string{
				"* ",
				"A ",
				"A tiež ",
				"A taktiež ",
				"A zároveň ",
			},
			background: []string{
				"Pozadie",
			},
			but: []string{
				"* ",
				"Ale ",
			},
			examples: []string{
				"Príklady",
			},
			feature: []string{
				"Požiadavka",
				"Funkcia",
				"Vlastnosť",
			},
			given: []string{
				"* ",
				"Pokiaľ ",
				"Za predpokladu ",
			},
			scenario: []string{
				"Scenár",
			},
			scenarioOutline: []string{
				"Náčrt Scenáru",
				"Náčrt Scenára",
				"Osnova Scenára",
			},
			then: []string{
				"* ",
				"Tak ",
				"Potom ",
			},
			when: []string{
				"* ",
				"Keď ",
				"Ak ",
			},
		},
	},
	"sl": &GherkinDialect{
		"sl", "Slovenian", "Slovenski", map[string][]string{
			and: []string{
				"In ",
				"Ter ",
			},
			background: []string{
				"Kontekst",
				"Osnova",
				"Ozadje",
			},
			but: []string{
				"Toda ",
				"Ampak ",
				"Vendar ",
			},
			examples: []string{
				"Primeri",
				"Scenariji",
			},
			feature: []string{
				"Funkcionalnost",
				"Funkcija",
				"Možnosti",
				"Moznosti",
				"Lastnost",
				"Značilnost",
			},
			given: []string{
				"Dano ",
				"Podano ",
				"Zaradi ",
				"Privzeto ",
			},
			scenario: []string{
				"Scenarij",
				"Primer",
			},
			scenarioOutline: []string{
				"Struktura scenarija",
				"Skica",
				"Koncept",
				"Oris scenarija",
				"Osnutek",
			},
			then: []string{
				"Nato ",
				"Potem ",
				"Takrat ",
			},
			when: []string{
				"Ko ",
				"Ce ",
				"Če ",
				"Kadar ",
			},
		},
	},
	"sr-Cyrl": &GherkinDialect{
		"sr-Cyrl", "Serbian", "Српски", map[string][]string{
			and: []string{
				"* ",
				"И ",
			},
			background: []string{
				"Контекст",
				"Основа",
				"Позадина",
			},
			but: []string{
				"* ",
				"Али ",
			},
			examples: []string{
				"Примери",
				"Сценарији",
			},
			feature: []string{
				"Функционалност",
				"Могућност",
				"Особина",
			},
			given: []string{
				"* ",
				"За дато ",
				"За дате ",
				"За дати ",
			},
			scenario: []string{
				"Сценарио",
				"Пример",
			},
			scenarioOutline: []string{
				"Структура сценарија",
				"Скица",
				"Концепт",
			},
			then: []string{
				"* ",
				"Онда ",
			},
			when: []string{
				"* ",
				"Када ",
				"Кад ",
			},
		},
	},
	"sr-Latn": &GherkinDialect{
		"sr-Latn", "Serbian (Latin)", "Srpski (Latinica)", map[string][]string{
			and: []string{
				"* ",
				"I ",
			},
			background: []string{
				"Kontekst",
				"Osnova",
				"Pozadina",
			},
			but: []string{
				"* ",
				"Ali ",
			},
			examples: []string{
				"Primeri",
				"Scenariji",
			},
			feature: []string{
				"Funkcionalnost",
				"Mogućnost",
				"Mogucnost",
				"Osobina",
			},
			given: []string{
				"* ",
				"Za dato ",
				"Za date ",
				"Za dati ",
			},
			scenario: []string{
				"Scenario",
				"Primer",
			},
			scenarioOutline: []string{
				"Struktura scenarija",
				"Skica",
				"Koncept",
			},
			then: []string{
				"* ",
				"Onda ",
			},
			when: []string{
				"* ",
				"Kada ",
				"Kad ",
			},
		},
	},
	"sv": &GherkinDialect{
		"sv", "Swedish", "Svenska", map[string][]string{
			and: []string{
				"* ",
				"Och ",
			},
			background: []string{
				"Bakgrund",
			},
			but: []string{
				"* ",
				"Men ",
			},
			examples: []string{
				"Exempel",
			},
			feature: []string{
				"Egenskap",
			},
			given: []string{
				"* ",
				"Givet ",
			},
			scenario: []string{
				"Scenario",
			},
			scenarioOutline: []string{
				"Abstrakt Scenario",
				"Scenariomall",
			},
			then: []string{
				"* ",
				"Så ",
			},
			when: []string{
				"* ",
				"När ",
			},
		},
	},
	"ta": &GherkinDialect{
		"ta", "Tamil", "தமிழ்", map[string][]string{
			and: []string{
				"* ",
				"மேலும்  ",
				"மற்றும் ",
			},
			background: []string{
				"பின்னணி",
			},
			but: []string{
				"* ",
				"ஆனால்  ",
			},
			examples: []string{
				"எடுத்துக்காட்டுகள்",
				"காட்சிகள்",
				" நிலைமைகளில்",
			},
			feature: []string{
				"அம்சம்",
				"வணிக தேவை",
				"திறன்",
			},
			given: []string{
				"* ",
				"கொடுக்கப்பட்ட ",
			},
			scenario: []string{
				"காட்சி",
			},
			scenarioOutline: []string{
				"காட்சி சுருக்கம்",
				"காட்சி வார்ப்புரு",
			},
			then: []string{
				"* ",
				"அப்பொழுது ",
			},
			when: []string{
				"* ",
				"எப்போது ",
			},
		},
	},
	"th": &GherkinDialect{
		"th", "Thai", "ไทย", map[string][]string{
			and: []string{
				"* ",
				"และ ",
			},
			background: []string{
				"แนวคิด",
			},
			but: []string{
				"* ",
				"แต่ ",
			},
			examples: []string{
				"ชุดของตัวอย่าง",
				"ชุดของเหตุการณ์",
			},
			feature: []string{
				"โครงหลัก",
				"ความต้องการทางธุรกิจ",
				"ความสามารถ",
			},
			given: []string{
				"* ",
				"กำหนดให้ ",
			},
			scenario: []string{
				"เหตุการณ์",
			},
			scenarioOutline: []string{
				"สรุปเหตุการณ์",
				"โครงสร้างของเหตุการณ์",
			},
			then: []string{
				"* ",
				"ดังนั้น ",
			},
			when: []string{
				"* ",
				"เมื่อ ",
			},
		},
	},
	"tl": &GherkinDialect{
		"tl", "Telugu", "తెలుగు", map[string][]string{
			and: []string{
				"* ",
				"మరియు ",
			},
			background: []string{
				"నేపథ్యం",
			},
			but: []string{
				"* ",
				"కాని ",
			},
			examples: []string{
				"ఉదాహరణలు",
			},
			feature: []string{
				"గుణము",
			},
			given: []string{
				"* ",
				"చెప్పబడినది ",
			},
			scenario: []string{
				"సన్నివేశం",
			},
			scenarioOutline: []string{
				"కథనం",
			},
			then: []string{
				"* ",
				"అప్పుడు ",
			},
			when: []string{
				"* ",
				"ఈ పరిస్థితిలో ",
			},
		},
	},
	"tlh": &GherkinDialect{
		"tlh", "Klingon", "tlhIngan", map[string][]string{
			and: []string{
				"* ",
				"'ej ",
				"latlh ",
			},
			background: []string{
				"mo'",
			},
			but: []string{
				"* ",
				"'ach ",
				"'a ",
			},
			examples: []string{
				"ghantoH",
				"lutmey",
			},
			feature: []string{
				"Qap",
				"Qu'meH 'ut",
				"perbogh",
				"poQbogh malja'",
				"laH",
			},
			given: []string{
				"* ",
				"ghu' noblu' ",
				"DaH ghu' bejlu' ",
			},
			scenario: []string{
				"lut",
			},
			scenarioOutline: []string{
				"lut chovnatlh",
			},
			then: []string{
				"* ",
				"vaj ",
			},
			when: []string{
				"* ",
				"qaSDI' ",
			},
		},
	},
	"tr": &GherkinDialect{
		"tr", "Turkish", "Türkçe", map[string][]string{
			and: []string{
				"* ",
				"Ve ",
			},
			background: []string{
				"Geçmiş",
			},
			but: []string{
				"* ",
				"Fakat ",
				"Ama ",
			},
			examples: []string{
				"Örnekler",
			},
			feature: []string{
				"Özellik",
			},
			given: []string{
				"* ",
				"Diyelim ki ",
			},
			scenario: []string{
				"Senaryo",
			},
			scenarioOutline: []string{
				"Senaryo taslağı",
			},
			then: []string{
				"* ",
				"O zaman ",
			},
			when: []string{
				"* ",
				"Eğer ki ",
			},
		},
	},
	"tt": &GherkinDialect{
		"tt", "Tatar", "Татарча", map[string][]string{
			and: []string{
				"* ",
				"Һәм ",
				"Вә ",
			},
			background: []string{
				"Кереш",
			},
			but: []string{
				"* ",
				"Ләкин ",
				"Әмма ",
			},
			examples: []string{
				"Үрнәкләр",
				"Мисаллар",
			},
			feature: []string{
				"Мөмкинлек",
				"Үзенчәлеклелек",
			},
			given: []string{
				"* ",
				"Әйтик ",
			},
			scenario: []string{
				"Сценарий",
			},
			scenarioOutline: []string{
				"Сценарийның төзелеше",
			},
			then: []string{
				"* ",
				"Нәтиҗәдә ",
			},
			when: []string{
				"* ",
				"Әгәр ",
			},
		},
	},
	"uk": &GherkinDialect{
		"uk", "Ukrainian", "Українська", map[string][]string{
			and: []string{
				"* ",
				"І ",
				"А також ",
				"Та ",
			},
			background: []string{
				"Передумова",
			},
			but: []string{
				"* ",
				"Але ",
			},
			examples: []string{
				"Приклади",
			},
			feature: []string{
				"Функціонал",
			},
			given: []string{
				"* ",
				"Припустимо ",
				"Припустимо, що ",
				"Нехай ",
				"Дано ",
			},
			scenario: []string{
				"Сценарій",
			},
			scenarioOutline: []string{
				"Структура сценарію",
			},
			then: []string{
				"* ",
				"То ",
				"Тоді ",
			},
			when: []string{
				"* ",
				"Якщо ",
				"Коли ",
			},
		},
	},
	"ur": &GherkinDialect{
		"ur", "Urdu", "اردو", map[string][]string{
			and: []string{
				"* ",
				"اور ",
			},
			background: []string{
				"پس منظر",
			},
			but: []string{
				"* ",
				"لیکن ",
			},
			examples: []string{
				"مثالیں",
			},
			feature: []string{
				"صلاحیت",
				"کاروبار کی ضرورت",
				"خصوصیت",
			},
			given: []string{
				"* ",
				"اگر ",
				"بالفرض ",
				"فرض کیا ",
			},
			scenario: []string{
				"منظرنامہ",
			},
			scenarioOutline: []string{
				"منظر نامے کا خاکہ",
			},
			then: []string{
				"* ",
				"پھر ",
				"تب ",
			},
			when: []string{
				"* ",
				"جب ",
			},
		},
	},
	"uz": &GherkinDialect{
		"uz", "Uzbek", "Узбекча", map[string][]string{
			and: []string{
				"* ",
				"Ва ",
			},
			background: []string{
				"Тарих",
			},
			but: []string{
				"* ",
				"Лекин ",
				"Бирок ",
				"Аммо ",
			},
			examples: []string{
				"Мисоллар",
			},
			feature: []string{
				"Функционал",
			},
			given: []string{
				"* ",
				"Агар ",
			},
			scenario: []string{
				"Сценарий",
			},
			scenarioOutline: []string{
				"Сценарий структураси",
			},
			then: []string{
				"* ",
				"Унда ",
			},
			when: []string{
				"* ",
				"Агар ",
			},
		},
	},
	"vi": &GherkinDialect{
		"vi", "Vietnamese", "Tiếng Việt", map[string][]string{
			and: []string{
				"* ",
				"Và ",
			},
			background: []string{
				"Bối cảnh",
			},
			but: []string{
				"* ",
				"Nhưng ",
			},
			examples: []string{
				"Dữ liệu",
			},
			feature: []string{
				"Tính năng",
			},
			given: []string{
				"* ",
				"Biết ",
				"Cho ",
			},
			scenario: []string{
				"Tình huống",
				"Kịch bản",
			},
			scenarioOutline: []string{
				"Khung tình huống",
				"Khung kịch bản",
			},
			then: []string{
				"* ",
				"Thì ",
			},
			when: []string{
				"* ",
				"Khi ",
			},
		},
	},
	"zh-CN": &GherkinDialect{
		"zh-CN", "Chinese simplified", "简体中文", map[string][]string{
			and: []string{
				"* ",
				"而且",
				"并且",
				"同时",
			},
			background: []string{
				"背景",
			},
			but: []string{
				"* ",
				"但是",
			},
			examples: []string{
				"例子",
			},
			feature: []string{
				"功能",
			},
			given: []string{
				"* ",
				"假如",
				"假设",
				"假定",
			},
			scenario: []string{
				"场景",
				"剧本",
			},
			scenarioOutline: []string{
				"场景大纲",
				"剧本大纲",
			},
			then: []string{
				"* ",
				"那么",
			},
			when: []string{
				"* ",
				"当",
			},
		},
	},
	"zh-TW": &GherkinDialect{
		"zh-TW", "Chinese traditional", "繁體中文", map[string][]string{
			and: []string{
				"* ",
				"而且",
				"並且",
				"同時",
			},
			background: []string{
				"背景",
			},
			but: []string{
				"* ",
				"但是",
			},
			examples: []string{
				"例子",
			},
			feature: []string{
				"功能",
			},
			given: []string{
				"* ",
				"假如",
				"假設",
				"假定",
			},
			scenario: []string{
				"場景",
				"劇本",
			},
			scenarioOutline: []string{
				"場景大綱",
				"劇本大綱",
			},
			then: []string{
				"* ",
				"那麼",
			},
			when: []string{
				"* ",
				"當",
			},
		},
	},
}
