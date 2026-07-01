"""
AETHER DATASET GENERATOR v6 — High Quality Bilingual (EN+GR)
  Strategy: Coherent, factually grounded, subject-specific entries.
  Each entry uses subject-aware facts, NOT random template combinations.
  Target: 512MB, ~40% raw / 60% Q&A, ordered interleave.
  
  Identity Strategy: Aether identity is present but not overwhelming (~2% frequency)
"""

import json
import random
import os

random.seed(42)

def pick(p):
    return random.choice(p)

def pick_subject(subjects, identity_weight=0.02):
    """
    Pick a subject with weighted probability.
    First subject (Aether) appears with reduced frequency.
    
    Args:
        subjects: List of subject dictionaries
        identity_weight: Probability of selecting Aether (default 2%)
    """
    if random.random() < identity_weight:
        # Select Aether (first subject)
        return subjects[0]
    else:
        # Select from other subjects uniformly
        return random.choice(subjects[1:])

def _smart_connector(connector: str, fact: str) -> str:
    """Connect a transition to a fact, preserving acronyms and proper nouns."""
    if len(fact) < 2:
        return f"{connector} {fact.lower()}"
    first_two = fact[:2]
    # Preserve true acronyms (2+ consecutive uppercase letters: GPS, DNA, CRISPR, fMRI)
    if first_two[0].isupper() and first_two[1].isupper():
        return f"{connector} {fact}"
    first_word = fact.split()[0] if fact.split() else ""
    stripped = first_word.rstrip(",.!?;:")
    stripped_no_poss = stripped.rstrip("'s") if stripped.endswith("'s") else stripped
    # Known proper nouns that should keep their capital after a connector
    proper_nouns = {
        # People
        "Stephen", "Niels", "Charles", "Edward", "Alexander", "Aristotle",
        "Plato", "Socrates", "Homer", "Darwin", "Newton", "Einstein",
        "Hawking", "Heisenberg", "Bohr", "Schrödinger", "Watson",
        "Crick", "Franklin", "Mendel", "Kepler", "Copernicus", "Galileo",
        "Turing", "Wallace", "Lamarck", "Linnaeus", "Pasteur", "Koch",
        "Fleming", "Jenner", "Watt", "Gutenberg", "Machiavelli", "Nakamoto",
        "Caesar", "Augustus", "Leonardo", "Michelangelo", "Raphael", "Satoshi",
        "Louis", "Marie", "Isaac", "Alan", "Francis", "Rosalind", "James",
        "Johannes", "Nikola", "Thomas", "Hippocrates", "Ptolemy", "Euclid",
        "Pythagoras", "Niccolò", "William", "Michael",
        # Acronyms and technical terms
        "AI", "C4", "CAM", "DNA", "GPS", "mRNA", "NFTs", "CRISPR", "CRISPR-Cas9",
        "MRI", "fMRI", "EEG", "PET", "LIGO", "IPCC", "NASA", "COVID", "MMR",
        "CRISPR", "ATP", "NADPH", "CO₂", "CO2",
        "Homo",
        # Places and nationalities
        "Athens", "Rome", "Greek", "Roman", "European", "Athenian", "Latin",
        # Named concepts and diseases
        "Renaissance", "Alzheimer", "Parkinson",
    }
    if stripped in proper_nouns or stripped_no_poss in proper_nouns:
        return f"{connector} {fact}"
    return f"{connector} {fact[0].lower() + fact[1:]}"

def _natural_join(facts: list, connectors: list, closers: list, name: str,
                  app: str, start_fact: bool = True) -> str:
    """Build a flowing paragraph from facts with varied structure."""
    if start_fact and len(facts) >= 2:
        parts = [facts[0]]
        remaining = facts[1:]
    else:
        parts = []
        remaining = facts
    for i, fact in enumerate(remaining):
        if i == len(remaining) - 1 and random.random() < 0.35:
            parts.append(pick(closers))
        else:
            parts.append(_smart_connector(pick(connectors), fact))
    if not any(c in parts[-1] for c in ["This is", "this is", "Ultimately", "All of"]):
        if random.random() < 0.55:
            parts.append(pick(closers))
    return " ".join(parts)

# ===================================================================
#  SUBJECT KNOWLEDGE BASE — each subject has real, coherent facts
# ===================================================================

SUBJECTS_EN = [
    {
        "name": "Aether AI model",
        "facts": [
            "Aether is a 51-million-parameter bilingual language model that speaks both English and Greek fluently.",
            "The model was created by Konpep, a developer passionate about making AI accessible to Greek speakers.",
            "Aether is based on the RWKV version 4 architecture, which offers linear complexity instead of quadratic attention.",
            "The architecture consists of 14 layers with a hidden size of 640 dimensions.",
            "RWKV models train like Transformers but run with O(T) linear complexity during inference.",
            "Aether was trained on 500MB of bilingual text, covering diverse topics in both English and Greek.",
            "The training data includes raw text, question-answer pairs, and multi-turn conversations.",
            "Aether uses a byte-level BPE tokenizer with an 8,192-token vocabulary, handling any Unicode language.",
            "The model can run on CPU without requiring a GPU, making it accessible for inference on standard computers.",
            "Aether represents an effort to bring high-quality AI language models to the Greek-speaking community.",
        ],
        "applications": [
            "Aether can be used for bilingual text generation, translation assistance, and conversational AI",
            "the model serves as a foundation for fine-tuning on specialized Greek or English tasks",
            "Aether demonstrates that efficient architectures like RWKV can deliver strong performance with fewer parameters",
        ]
    },
    {
        "name": "black holes",
        "facts": [
            "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape.",
            "Black holes form when massive stars collapse under their own gravity at the end of their lives.",
            "The boundary of a black hole is called the event horizon — once crossed, there is no return.",
            "At the center of a black hole lies a singularity, a point of infinite density.",
            "Supermassive black holes, millions to billions of times the mass of the Sun, exist at the centers of most galaxies.",
            "Stephen Hawking proposed that black holes slowly emit radiation, now called Hawking radiation.",
            "The first image of a black hole was captured in 2019 by the Event Horizon Telescope.",
            "Black holes can merge, producing gravitational waves detectable by instruments like LIGO.",
            "Time passes more slowly near a black hole due to gravitational time dilation.",
            "Black holes are classified as stellar, intermediate, supermassive, or primordial based on their mass.",
        ],
        "applications": [
            "studying black holes helps us test the limits of general relativity",
            "understanding black holes sheds light on the early universe and galaxy formation",
            "gravitational wave astronomy, enabled by black hole mergers, is a new way to observe the universe",
        ]
    },
    {
        "name": "quantum mechanics",
        "facts": [
            "Quantum mechanics describes the behavior of matter and energy at the smallest scales.",
            "The wave-particle duality principle states that particles like electrons exhibit both wave and particle properties.",
            "Heisenberg's uncertainty principle states that you cannot know both the position and momentum of a particle precisely.",
            "Quantum superposition allows particles to exist in multiple states simultaneously until measured.",
            "Quantum entanglement links two particles so that measuring one instantly affects the other, regardless of distance.",
            "The Schrödinger equation describes how quantum states evolve over time.",
            "Quantum tunneling allows particles to pass through barriers they classically should not be able to cross.",
            "Quantum mechanics underpins technologies like lasers, transistors, and MRI machines.",
            "Unlike classical physics, quantum mechanics is fundamentally probabilistic.",
            "Niels Bohr, Werner Heisenberg, and Erwin Schrödinger were pioneering figures in quantum theory.",
        ],
        "applications": [
            "quantum mechanics is the foundation of modern electronics and computing",
            "quantum cryptography uses quantum principles to create theoretically unbreakable encryption",
            "quantum computers leverage superposition and entanglement to solve certain problems exponentially faster",
        ]
    },
    {
        "name": "DNA",
        "facts": [
            "DNA, or deoxyribonucleic acid, carries the genetic instructions for all living organisms.",
            "The DNA double helix structure was discovered by Watson and Crick in 1953, building on Rosalind Franklin's X-ray work.",
            "DNA is composed of four nucleotide bases: adenine (A), thymine (T), cytosine (C), and guanine (G).",
            "The human genome contains approximately 3 billion base pairs encoding around 20,000 genes.",
            "DNA replication ensures that each new cell receives an accurate copy of the genetic information.",
            "Mutations in DNA can lead to changes in protein function, sometimes causing diseases like cancer.",
            "DNA is found in the nucleus of cells, as well as in mitochondria.",
            "Gene expression involves transcribing DNA into RNA, which is then translated into proteins.",
            "CRISPR-Cas9 is a revolutionary gene editing tool that allows precise modifications to DNA sequences.",
            "DNA profiling, or fingerprinting, is used in forensics to identify individuals with high accuracy.",
        ],
        "applications": [
            "understanding DNA enables advances in medicine, including gene therapy and personalized treatment",
            "DNA sequencing has transformed our understanding of evolution and species relationships",
            "agricultural biotechnology uses DNA knowledge to develop disease-resistant and higher-yield crops",
        ]
    },
    {
        "name": "photosynthesis",
        "facts": [
            "Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight into chemical energy.",
            "The overall equation is: CO2 + H2O + light → glucose + O2.",
            "Photosynthesis takes place primarily in the chloroplasts of plant cells.",
            "Chlorophyll, the green pigment in leaves, absorbs red and blue light but reflects green light.",
            "Photosynthesis occurs in two stages: the light-dependent reactions and the Calvin cycle.",
            "During the light-dependent reactions, water is split and oxygen is released as a byproduct.",
            "The Calvin cycle uses CO2 and ATP to produce glucose in the stroma of chloroplasts.",
            "Photosynthesis is the foundation of almost all food chains on Earth.",
            "Factors like light intensity, CO2 concentration, and temperature affect the rate of photosynthesis.",
            "C4 and CAM plants have evolved specialized forms of photosynthesis to minimize water loss.",
        ],
        "applications": [
            "understanding photosynthesis guides the development of crops with greater yields",
            "artificial photosynthesis is being researched as a way to produce clean energy from sunlight",
            "photosynthesis research informs models of global carbon cycling and climate change",
        ]
    },
    {
        "name": "the water cycle",
        "facts": [
            "The water cycle, or hydrological cycle, describes the continuous movement of water through Earth's systems.",
            "Key stages include evaporation, condensation, precipitation, and runoff.",
            "The Sun's energy drives evaporation, turning liquid water into water vapor.",
            "Water vapor rises, cools, and condenses around tiny particles to form clouds.",
            "Precipitation returns water to Earth's surface as rain, snow, sleet, or hail.",
            "Runoff flows into rivers and oceans, while some water infiltrates the soil and replenishes groundwater.",
            "Transpiration from plants is a significant source of water vapor in the atmosphere.",
            "The water cycle regulates Earth's temperature and distributes freshwater across the planet.",
            "Human activities like deforestation and urbanization can disrupt the natural water cycle.",
            "Climate change is altering precipitation patterns and increasing the frequency of droughts and floods.",
        ],
        "applications": [
            "understanding the water cycle is essential for managing freshwater resources",
            "hydrological models based on the water cycle help predict floods and droughts",
            "irrigation and water conservation strategies rely on knowledge of the water cycle",
        ]
    },
    {
        "name": "evolution",
        "facts": [
            "Evolution is the change in the inherited characteristics of populations over successive generations.",
            "Charles Darwin proposed the theory of natural selection in 'On the Origin of Species' in 1859.",
            "Natural selection favors traits that improve an organism's chances of survival and reproduction.",
            "Genetic mutations are the ultimate source of new variation in populations.",
            "Speciation occurs when populations become reproductively isolated and diverge over time.",
            "The fossil record provides direct evidence of evolutionary change across geological time.",
            "All life on Earth shares a common ancestor, as supported by molecular biology and genetics.",
            "Convergent evolution occurs when unrelated species independently evolve similar traits.",
            "Human evolution involved several hominin species, with Homo sapiens emerging around 300,000 years ago.",
            "Evolutionary biology informs medicine, agriculture, and our understanding of biodiversity.",
        ],
        "applications": [
            "evolutionary principles explain the development of antibiotic resistance in bacteria",
            "phylogenetics uses evolutionary data to classify organisms and reconstruct ancestral relationships",
            "conservation biology applies evolutionary thinking to protect endangered species",
        ]
    },
    {
        "name": "the human brain",
        "facts": [
            "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
            "The brain is divided into regions with specialized functions: the frontal lobe handles reasoning, the occipital lobe processes vision.",
            "The cerebral cortex is the outer layer responsible for higher-order thinking, language, and consciousness.",
            "The limbic system, including the amygdala and hippocampus, regulates emotion and memory.",
            "The brain communicates via electrical signals (action potentials) and chemical neurotransmitters.",
            "Neuroplasticity allows the brain to reorganize and form new connections throughout life.",
            "During sleep, the brain consolidates memories and clears toxic waste products.",
            "The brain consumes about 20% of the body's energy despite being only 2% of body weight.",
            "Disorders like Alzheimer's, Parkinson's, and depression involve disruptions in brain structure and chemistry.",
            "Neuroimaging techniques like fMRI and EEG allow scientists to observe brain activity in real time.",
        ],
        "applications": [
            "neuroscience research drives the development of treatments for neurological and psychiatric disorders",
            "understanding the brain informs the design of artificial neural networks in AI",
            "cognitive neuroscience insights are applied in education, therapy, and human-computer interaction",
        ]
    },
    {
        "name": "artificial intelligence",
        "facts": [
            "Artificial intelligence is the simulation of human intelligence processes by machines, particularly computers.",
            "AI encompasses machine learning, deep learning, natural language processing, and computer vision.",
            "Machine learning algorithms improve through exposure to data without being explicitly programmed.",
            "Deep learning uses multi-layered neural networks to recognize patterns in large datasets.",
            "Natural language processing enables computers to understand, generate, and translate human language.",
            "AI has achieved superhuman performance in games like chess and Go, and in tasks like image recognition.",
            "Large language models like GPT are trained on vast text corpora and can generate coherent text.",
            "AI raises ethical questions about bias, privacy, job displacement, and autonomous decision-making.",
            "Autonomous vehicles use AI to perceive their environment and make real-time driving decisions.",
            "AI is transforming industries including healthcare, finance, logistics, and scientific research.",
        ],
        "applications": [
            "AI-powered diagnostic tools can detect diseases like cancer in medical images with high accuracy",
            "recommendation systems driven by AI personalize content on streaming and e-commerce platforms",
            "AI accelerates drug discovery by predicting molecular structures and simulating interactions",
        ]
    },
    {
        "name": "climate change",
        "facts": [
            "Climate change refers to long-term shifts in global temperatures and weather patterns, largely driven by human activities.",
            "The burning of fossil fuels releases CO2 and other greenhouse gases that trap heat in the atmosphere.",
            "Since the industrial revolution, global average temperatures have risen by approximately 1.1°C.",
            "The IPCC warns that exceeding 1.5°C of warming will significantly increase extreme weather events.",
            "Melting polar ice and glaciers are causing sea levels to rise, threatening coastal communities.",
            "Ocean acidification, caused by absorbed CO2, threatens marine ecosystems including coral reefs.",
            "Deforestation reduces Earth's capacity to absorb CO2, accelerating warming.",
            "Renewable energy, energy efficiency, and carbon capture are key strategies for mitigation.",
            "Climate change disproportionately affects vulnerable populations in developing countries.",
            "International agreements like the Paris Accord aim to coordinate global action on emissions reduction.",
        ],
        "applications": [
            "climate models help scientists project future temperature and precipitation changes",
            "understanding climate change informs policy on renewable energy and carbon pricing",
            "adaptation strategies help communities prepare for floods, droughts, and heat waves",
        ]
    },
    {
        "name": "the Roman Empire",
        "facts": [
            "The Roman Empire was one of the largest empires in history, at its peak spanning from Britain to Mesopotamia.",
            "It began under Augustus Caesar in 27 BCE and the Western Roman Empire fell in 476 CE.",
            "Rome's republican system of government influenced modern democracies and legal systems.",
            "Latin, the language of Rome, evolved into the Romance languages including Spanish, French, and Italian.",
            "The Romans built an extensive road network, aqueducts, and architectural monuments that still stand today.",
            "Roman law established principles such as innocent until proven guilty that underpin modern legal codes.",
            "The army was the backbone of Roman power, consisting of disciplined legions with sophisticated tactics.",
            "The Colosseum, completed in 80 CE, held up to 80,000 spectators for gladiatorial contests.",
            "Rome adopted and adapted Greek culture, philosophy, and religion throughout its history.",
            "The spread of Christianity was significantly shaped by the Roman Empire's infrastructure and reach.",
        ],
        "applications": [
            "Roman engineering principles are still referenced in modern architecture and civil engineering",
            "Roman legal concepts form the basis of legal systems across Europe and Latin America",
            "the study of Rome illuminates how complex societies rise, expand, and eventually decline",
        ]
    },
    {
        "name": "machine learning",
        "facts": [
            "Machine learning is a branch of AI where algorithms learn from data to make predictions or decisions.",
            "Supervised learning uses labeled training data to teach a model to map inputs to outputs.",
            "Unsupervised learning finds hidden patterns in unlabeled data through clustering and dimensionality reduction.",
            "Reinforcement learning trains agents by rewarding desired behaviors in an environment.",
            "Overfitting occurs when a model learns the training data too well and fails to generalize.",
            "Cross-validation is a technique used to assess how well a model will generalize to new data.",
            "Gradient descent is the optimization algorithm used to minimize loss functions during training.",
            "Feature engineering involves selecting and transforming input variables to improve model performance.",
            "Neural networks, inspired by the brain, are the basis of modern deep learning systems.",
            "Common applications include spam detection, image classification, fraud detection, and recommendation systems.",
        ],
        "applications": [
            "machine learning enables personalized medicine by predicting patient outcomes from health data",
            "financial institutions use ML models to detect fraudulent transactions in real time",
            "language models trained via ML can translate, summarize, and generate human-like text",
        ]
    },
    {
        "name": "democracy",
        "facts": [
            "Democracy is a system of government in which power is vested in the people, exercised directly or through elected representatives.",
            "Ancient Athens is considered the birthplace of democracy, introducing direct participation in the 5th century BCE.",
            "Modern representative democracies elect officials through free and fair elections.",
            "Key principles include freedom of speech, rule of law, separation of powers, and protection of minority rights.",
            "The Magna Carta (1215) was an early milestone in limiting the power of rulers and establishing rights.",
            "Liberal democracy combines democratic elections with the protection of individual liberties.",
            "Authoritarian regimes often hold elections but undermine democratic norms through corruption and suppression.",
            "Voter turnout, civic education, and free press are considered vital for healthy democracies.",
            "Digital technology has created new opportunities and threats for democratic participation.",
            "The spread of democracy worldwide accelerated after World War II and again after the Cold War.",
        ],
        "applications": [
            "studying democracy helps identify the conditions under which free societies flourish or decline",
            "democratic institutions provide frameworks for peaceful transfer of power and conflict resolution",
            "civic education rooted in democratic values prepares citizens to participate in governance",
        ]
    },
    {
        "name": "the theory of relativity",
        "facts": [
            "Einstein's theory of relativity consists of special relativity (1905) and general relativity (1915).",
            "Special relativity states that the laws of physics are the same for all non-accelerating observers.",
            "The famous equation E=mc² shows that mass and energy are interchangeable.",
            "Time dilation means that a moving clock ticks slower relative to a stationary one.",
            "Length contraction means that objects moving at high speed are shorter in the direction of motion.",
            "General relativity describes gravity not as a force but as the curvature of spacetime caused by mass.",
            "Light bends around massive objects, a phenomenon confirmed during the 1919 solar eclipse.",
            "GPS satellites must correct for relativistic time effects to provide accurate location data.",
            "Black holes and gravitational waves are direct consequences of general relativity.",
            "Relativity has been confirmed by countless experiments and remains one of physics' most successful theories.",
        ],
        "applications": [
            "GPS technology relies on relativistic corrections to remain accurate",
            "understanding relativity is essential for space navigation and satellite communications",
            "general relativity provides the framework for cosmology and our understanding of the universe's structure",
        ]
    },
    {
        "name": "the Renaissance",
        "facts": [
            "The Renaissance was a cultural and intellectual movement that began in Italy in the 14th century and spread across Europe.",
            "It marked a rebirth of interest in classical Greek and Roman art, philosophy, and science.",
            "Key figures include Leonardo da Vinci, Michelangelo, Raphael, Galileo, and Machiavelli.",
            "The invention of the printing press by Gutenberg around 1440 accelerated the spread of Renaissance ideas.",
            "Renaissance art emphasized perspective, realism, and the human form, departing from medieval styles.",
            "Humanism, a philosophy centered on human potential and achievement, was central to Renaissance thought.",
            "Scientific inquiry flourished, laying the groundwork for the Scientific Revolution.",
            "The Sistine Chapel ceiling, painted by Michelangelo, is one of the most celebrated artworks in history.",
            "The Renaissance transformed European society, politics, and culture, leading to the modern era.",
            "It contributed to the Age of Exploration as Europeans sought new knowledge and trade routes.",
        ],
        "applications": [
            "Renaissance ideals of humanism continue to influence modern education and liberal arts curricula",
            "the rediscovery of classical texts during the Renaissance shaped Western philosophy and science",
            "Renaissance art techniques like perspective remain foundational in visual arts education",
        ]
    },
    {
        "name": "blockchain",
        "facts": [
            "A blockchain is a distributed ledger that records transactions across many computers in a tamper-resistant way.",
            "Each block contains a cryptographic hash of the previous block, transaction data, and a timestamp.",
            "Bitcoin, introduced in 2008 by the pseudonymous Satoshi Nakamoto, was the first blockchain application.",
            "Decentralization means no single entity controls the blockchain; consensus mechanisms validate transactions.",
            "Proof of Work requires computational effort to add blocks, while Proof of Stake relies on staked currency.",
            "Smart contracts are self-executing programs stored on blockchains that automate agreements.",
            "Ethereum introduced programmable smart contracts, enabling decentralized applications (dApps).",
            "NFTs (non-fungible tokens) use blockchain to certify the ownership of unique digital assets.",
            "Blockchain has potential applications in supply chain transparency, voting systems, and healthcare records.",
            "Scalability, energy consumption, and regulatory uncertainty remain major challenges for blockchain adoption.",
        ],
        "applications": [
            "blockchain enables peer-to-peer financial transactions without intermediaries like banks",
            "supply chain management uses blockchain to ensure the authenticity and traceability of goods",
            "decentralized finance (DeFi) platforms use blockchain to offer financial services without traditional institutions",
        ]
    },
    {
        "name": "neuroscience",
        "facts": [
            "Neuroscience is the scientific study of the nervous system, including the brain, spinal cord, and peripheral nerves.",
            "The brain contains approximately 86 billion neurons, each connected to thousands of others via synapses.",
            "Action potentials are electrical signals that neurons use to communicate with each other.",
            "Neurotransmitters like dopamine, serotonin, and glutamate mediate communication at synapses.",
            "Neuroplasticity is the brain's ability to reorganize itself by forming new neural connections.",
            "The hippocampus plays a crucial role in forming and consolidating long-term memories.",
            "Sleep is essential for memory consolidation and clearing neurotoxic waste from the brain.",
            "Imaging techniques like fMRI, EEG, and PET allow scientists to observe brain activity non-invasively.",
            "Disorders such as Alzheimer's, depression, schizophrenia, and epilepsy involve disrupted brain function.",
            "Optogenetics is a technique using light to control specific neurons, revolutionizing brain research.",
        ],
        "applications": [
            "neuroscience research informs new treatments for Alzheimer's, Parkinson's, and psychiatric disorders",
            "brain-computer interfaces developed through neuroscience can restore movement to paralyzed patients",
            "insights from neuroscience improve learning strategies and educational methodologies",
        ]
    },
    {
        "name": "vaccines",
        "facts": [
            "Vaccines stimulate the immune system to recognize and fight specific pathogens without causing disease.",
            "Edward Jenner pioneered vaccination in 1796 using cowpox material to protect against smallpox.",
            "Vaccines contain antigens such as weakened pathogens, inactivated viruses, or protein subunits.",
            "mRNA vaccines, like those developed for COVID-19, instruct cells to produce a protein that triggers immunity.",
            "Herd immunity is achieved when a sufficient percentage of a population is immune, protecting even the unvaccinated.",
            "Smallpox was eradicated in 1980 through a global vaccination campaign — the only human disease eradicated to date.",
            "The measles, mumps, and rubella (MMR) vaccine has dramatically reduced these once-common childhood diseases.",
            "Vaccine hesitancy, driven by misinformation, remains a significant public health challenge.",
            "Cold chain logistics — maintaining vaccines at the right temperature — is critical for effectiveness.",
            "The rapid development of COVID-19 vaccines demonstrated the potential of mRNA technology for future vaccines.",
        ],
        "applications": [
            "vaccines have prevented hundreds of millions of deaths and remain the most cost-effective public health intervention",
            "new vaccine platforms developed during COVID-19 are being applied to cancer, HIV, and malaria research",
            "routine childhood vaccination programs have eliminated polio from most of the world",
        ]
    },
    {
        "name": "ancient Greece",
        "facts": [
            "Ancient Greece was a civilization flourishing from around the 8th to 4th century BCE, centered in the Aegean region.",
            "The city-state, or polis, was the fundamental political unit, with Athens and Sparta being the most prominent.",
            "Athens pioneered democratic governance, where citizens could vote on laws and policies.",
            "Greek philosophers like Socrates, Plato, and Aristotle laid the foundations of Western philosophy.",
            "The Olympic Games, first held in 776 BCE at Olympia, brought city-states together in peaceful competition.",
            "Greek literature includes the epic poems of Homer — the Iliad and the Odyssey.",
            "Ancient Greeks made foundational contributions to mathematics, medicine, astronomy, and science.",
            "Alexander the Great spread Greek culture across a vast empire from Egypt to India, creating the Hellenistic world.",
            "Greek mythology featured gods like Zeus, Athena, and Apollo, explaining natural and human phenomena.",
            "The Parthenon, built on the Athenian Acropolis in the 5th century BCE, remains an architectural masterpiece.",
        ],
        "applications": [
            "Greek philosophy underpins modern ethics, logic, and scientific method",
            "the democratic ideals of Athens directly influenced the design of modern governance systems",
            "Greek literature and mythology continue to inspire art, literature, and culture worldwide",
        ]
    },
    {
        "name": "renewable energy",
        "facts": [
            "Renewable energy comes from naturally replenishing sources such as solar, wind, water, geothermal, and biomass.",
            "Solar photovoltaic panels convert sunlight directly into electricity through the photovoltaic effect.",
            "Wind turbines generate electricity by converting the kinetic energy of wind into rotational motion.",
            "Hydropower, the largest source of renewable electricity globally, uses flowing water to spin turbines.",
            "Offshore wind farms are expanding rapidly due to stronger and more consistent winds over the ocean.",
            "Renewable energy costs have fallen dramatically — solar power costs have dropped over 90% in the past decade.",
            "Energy storage, particularly lithium-ion batteries, is key to managing the intermittency of solar and wind power.",
            "Geothermal energy taps heat from the Earth's interior and is particularly abundant in volcanic regions.",
            "Transitioning to renewables is essential for reducing greenhouse gas emissions and meeting climate goals.",
            "By 2023, renewable energy sources accounted for over 30% of global electricity generation.",
        ],
        "applications": [
            "widespread adoption of renewables reduces dependence on fossil fuels and cuts carbon emissions",
            "off-grid solar systems provide electricity to remote communities without access to centralized power grids",
            "green hydrogen, produced using renewable electricity, is emerging as a clean fuel for industry and transport",
        ]
    },
    {
        "name": "the immune system",
        "facts": [
            "The immune system is the body's defense network against pathogens such as bacteria, viruses, and parasites.",
            "The innate immune system provides immediate, non-specific defense through physical barriers and inflammation.",
            "The adaptive immune system mounts a targeted response using T cells and B cells after recognizing specific antigens.",
            "Antibodies are proteins produced by B cells that bind to and neutralize specific pathogens.",
            "Vaccines work by training the adaptive immune system to recognize pathogens before infection occurs.",
            "Autoimmune diseases occur when the immune system mistakenly attacks the body's own tissues.",
            "The thymus is where T cells mature and are educated to distinguish self from non-self.",
            "Cytokines are signaling molecules that coordinate immune responses and regulate inflammation.",
            "Immunosuppressive drugs are used after organ transplants to prevent the immune system from rejecting the new organ.",
            "Immunotherapy harnesses the immune system to fight cancer, representing a major advance in oncology.",
        ],
        "applications": [
            "understanding the immune system drives development of new vaccines and treatments for infections",
            "immunotherapy has transformed cancer treatment, enabling durable remissions in some previously fatal cancers",
            "research into autoimmune diseases like rheumatoid arthritis and lupus is improving patients' quality of life",
        ]
    },
]

SUBJECTS_GR = [
    {
        "name": "το μοντέλο Aether",
        "facts": [
            "Το Aether είναι ένα δίγλωσσο γλωσσικό μοντέλο 51 εκατομμυρίων παραμέτρων που μιλάει Ελληνικά και Αγγλικά άπταιστα.",
            "Το μοντέλο δημιουργήθηκε από τον Konpep, έναν προγραμματιστή με πάθος για την προσβασιμότητα της ΤΝ στους Έλληνες.",
            "Το Aether βασίζεται στην αρχιτεκτονική RWKV έκδοση 4, που προσφέρει γραμμική πολυπλοκότητα αντί τετραγωνικής προσοχής.",
            "Η αρχιτεκτονική αποτελείται από 14 επίπεδα με κρυφό μέγεθος 640 διαστάσεων.",
            "Τα μοντέλα RWKV εκπαιδεύονται σαν Transformers αλλά τρέχουν με γραμμική πολυπλοκότητα O(T) κατά την εκτέλεση.",
            "Το Aether εκπαιδεύτηκε σε 500MB δίγλωσσου κειμένου, καλύπτοντας ποικίλα θέματα στα Ελληνικά και Αγγλικά.",
            "Τα δεδομένα εκπαίδευσης περιλαμβάνουν ανοιχτό κείμενο, ζεύγη ερώτησης-απάντησης και συνομιλίες πολλαπλών γύρων.",
            "Το Aether χρησιμοποιεί έναν tokenizer BPE επιπέδου byte με λεξιλόγιο 8.192 tokens, διαχειριζόμενο οποιαδήποτε γλώσσα Unicode.",
            "Το μοντέλο μπορεί να τρέξει σε CPU χωρίς να απαιτεί GPU, καθιστώντας το προσβάσιμο για εκτέλεση σε κανονικούς υπολογιστές.",
            "Το Aether αντιπροσωπεύει μια προσπάθεια να φέρει υψηλής ποιότητας γλωσσικά μοντέλα ΤΝ στην ελληνόφωνη κοινότητα.",
        ],
        "applications": [
            "το Aether μπορεί να χρησιμοποιηθεί για δίγλωσση παραγωγή κειμένου, βοήθεια μετάφρασης και συνομιλιακή ΤΝ",
            "το μοντέλο λειτουργεί ως βάση για fine-tuning σε εξειδικευμένες ελληνικές ή αγγλικές εργασίες",
            "το Aether αποδεικνύει ότι αποδοτικές αρχιτεκτονικές όπως το RWKV μπορούν να προσφέρουν ισχυρή απόδοση με λιγότερες παραμέτρους",
        ]
    },
    {
        "name": "η κβαντομηχανική",
        "facts": [
            "Η κβαντομηχανική περιγράφει τη συμπεριφορά της ύλης και της ενέργειας σε ατομικές και υποατομικές κλίμακες.",
            "Η αρχή της αβεβαιότητας του Heisenberg δηλώνει ότι δεν μπορούμε να γνωρίζουμε ταυτόχρονα με ακρίβεια τη θέση και την ορμή ενός σωματιδίου.",
            "Η κυματική-σωματιδιακή δυαλικότητα υποδηλώνει ότι τα ηλεκτρόνια και τα φωτόνια συμπεριφέρονται τόσο ως κύματα όσο και ως σωματίδια.",
            "Η κβαντική υπέρθεση επιτρέπει σε ένα σωματίδιο να βρίσκεται ταυτόχρονα σε πολλές καταστάσεις μέχρι να μετρηθεί.",
            "Η κβαντική διεμπλοκή συνδέει δύο σωματίδια έτσι ώστε η μέτρηση του ενός να επηρεάζει αμέσως το άλλο.",
            "Η εξίσωση Schrödinger περιγράφει πώς εξελίσσεται μια κβαντική κατάσταση με τον χρόνο.",
            "Η κβαντική σήραγγα επιτρέπει σε σωματίδια να διέρχονται από εμπόδια που κλασικά δεν θα μπορούσαν να ξεπεράσουν.",
            "Η κβαντομηχανική αποτελεί τη βάση τεχνολογιών όπως τα laser, τα τρανζίστορ και τα MRI.",
            "Η κβαντική υπολογιστική χρησιμοποιεί κβαντικά φαινόμενα για να λύνει ορισμένα προβλήματα εκθετικά πιο γρήγορα.",
            "Πρωτοπόρες μορφές της κβαντικής θεωρίας ήταν οι Bohr, Heisenberg και Schrödinger.",
        ],
        "applications": [
            "η κβαντομηχανική υπόκειται στην αρχή λειτουργίας σύγχρονων ηλεκτρονικών συσκευών και ημιαγωγών",
            "η κβαντική κρυπτογραφία χρησιμοποιεί κβαντικές αρχές για θεωρητικά αδύνατη να σπαστεί κρυπτογράφηση",
            "οι κβαντικοί υπολογιστές αποτελούν το επόμενο σύνορο στην επεξεργαστική ισχύ",
        ]
    },
    {
        "name": "η τεχνητή νοημοσύνη",
        "facts": [
            "Η τεχνητή νοημοσύνη αποσκοπεί στη δημιουργία μηχανών που μπορούν να εκτελούν εργασίες που κανονικά απαιτούν ανθρώπινη νοημοσύνη.",
            "Η μηχανική μάθηση επιτρέπει σε αλγορίθμους να βελτιώνονται μέσα από την εμπειρία χωρίς ρητό προγραμματισμό.",
            "Τα βαθιά νευρωνικά δίκτυα έχουν επιτύχει εντυπωσιακά αποτελέσματα σε αναγνώριση εικόνας, ομιλίας και κειμένου.",
            "Τα μεγάλα γλωσσικά μοντέλα εκπαιδεύονται σε τεράστιες ποσότητες κειμένου και μπορούν να παράγουν συνεκτικό κείμενο.",
            "Η τεχνητή νοημοσύνη έχει ξεπεράσει ανθρώπινες επιδόσεις στο σκάκι, στο Go και σε εργασίες ιατρικής διάγνωσης.",
            "Τα αυτόνομα οχήματα χρησιμοποιούν τεχνητή νοημοσύνη για αντίληψη του περιβάλλοντος και λήψη αποφάσεων.",
            "Η τεχνητή νοημοσύνη εγείρει ηθικά ζητήματα σχετικά με προκατάληψη, ιδιωτικότητα και αυτόματες αποφάσεις.",
            "Η επεξεργασία φυσικής γλώσσας επιτρέπει στους υπολογιστές να κατανοούν, να παράγουν και να μεταφράζουν ανθρώπινη γλώσσα.",
            "Η ΤΝ μεταμορφώνει τομείς όπως η υγεία, η εκπαίδευση, τα οικονομικά και η επιστημονική έρευνα.",
            "Το μέλλον της ΤΝ εξαρτάται από ισορροπία μεταξύ ικανότητας, ασφάλειας και ηθικής χρήσης.",
        ],
        "applications": [
            "τα διαγνωστικά εργαλεία τεχνητής νοημοσύνης εντοπίζουν ασθένειες όπως ο καρκίνος σε ιατρικές εικόνες",
            "τα συστήματα συστάσεων τεχνητής νοημοσύνης εξατομικεύουν περιεχόμενο σε πλατφόρμες ροής",
            "η τεχνητή νοημοσύνη επιταχύνει την ανακάλυψη νέων φαρμάκων και θεραπειών",
        ]
    },
    {
        "name": "η αρχαία Ελλάδα",
        "facts": [
            "Η αρχαία Ελλάδα ήταν πολιτισμός που άνθισε από τον 8ο έως τον 4ο αιώνα π.Χ. στην Αιγαιακή περιοχή.",
            "Η βασική πολιτική μονάδα ήταν η πόλη-κράτος ή πόλις, με σημαντικότερες την Αθήνα και τη Σπάρτη.",
            "Η Αθήνα είναι η γενέτειρα της δημοκρατίας, όπου οι πολίτες μπορούσαν να ψηφίζουν στη Εκκλησία του Δήμου.",
            "Οι φιλόσοφοι Σωκράτης, Πλάτωνας και Αριστοτέλης θεμελίωσαν τη δυτική φιλοσοφία.",
            "Οι Ολυμπιακοί Αγώνες, που ξεκίνησαν το 776 π.Χ. στην Ολυμπία, ένωναν τις πόλεις-κράτη.",
            "Η ελληνική λογοτεχνία περιλαμβάνει τα έπη του Ομήρου — Ιλιάδα και Οδύσσεια.",
            "Οι αρχαίοι Έλληνες έκαναν θεμελιώδεις συνεισφορές στα μαθηματικά, την ιατρική, την αστρονομία και τις επιστήμες.",
            "Ο Μέγας Αλέξανδρος εξαπλώθηκε ελληνικός πολιτισμός σε μια τεράστια αυτοκρατορία από την Αίγυπτο ως την Ινδία.",
            "Η ελληνική μυθολογία με θεούς όπως ο Δίας, η Αθηνά και ο Απόλλωνας εξηγούσε φυσικά και ανθρώπινα φαινόμενα.",
            "Ο Παρθενώνας στην Ακρόπολη των Αθηνών, χτισμένος τον 5ο αιώνα π.Χ., παραμένει αρχιτεκτονικό αριστούργημα.",
        ],
        "applications": [
            "η ελληνική φιλοσοφία υποστηρίζει τη σύγχρονη ηθική, τη λογική και την επιστημονική μέθοδο",
            "τα δημοκρατικά ιδεώδη της Αθήνας επηρέασαν άμεσα τα σύγχρονα συστήματα διακυβέρνησης",
            "η ελληνική λογοτεχνία και μυθολογία συνεχίζουν να εμπνέουν τέχνες και πολιτισμό παγκοσμίως",
        ]
    },
    {
        "name": "η κλιματική αλλαγή",
        "facts": [
            "Η κλιματική αλλαγή αναφέρεται στις μακροχρόνιες μεταβολές της θερμοκρασίας και των καιρικών προτύπων, κυρίως λόγω ανθρώπινων δραστηριοτήτων.",
            "Η καύση ορυκτών καυσίμων απελευθερώνει CO2 και άλλα αέρια θερμοκηπίου που παγιδεύουν θερμότητα στην ατμόσφαιρα.",
            "Από τη βιομηχανική επανάσταση, η μέση παγκόσμια θερμοκρασία έχει αυξηθεί περίπου 1,1°C.",
            "Η IPCC προειδοποιεί ότι η υπέρβαση του 1,5°C θέρμανσης θα αυξήσει σημαντικά τα ακραία καιρικά φαινόμενα.",
            "Η τήξη πολικών πάγων και παγετώνων προκαλεί άνοδο της στάθμης της θάλασσας, απειλώντας παράκτιες κοινότητες.",
            "Η οξίνιση των ωκεανών, που προκαλείται από την απορρόφηση CO2, απειλεί τα θαλάσσια οικοσυστήματα.",
            "Η αποψίλωση των δασών μειώνει την ικανότητα της Γης να απορροφά CO2.",
            "Οι ανανεώσιμες πηγές ενέργειας, η ενεργειακή απόδοση και η δέσμευση άνθρακα είναι βασικές στρατηγικές μετριασμού.",
            "Η κλιματική αλλαγή επηρεάζει δυσανάλογα τις ευάλωτες πληθυσμιακές ομάδες στις αναπτυσσόμενες χώρες.",
            "Διεθνείς συμφωνίες όπως η Συμφωνία του Παρισιού στοχεύουν στο συντονισμό της παγκόσμιας δράσης για μείωση εκπομπών.",
        ],
        "applications": [
            "τα κλιματικά μοντέλα βοηθούν τους επιστήμονες να προβλέπουν μελλοντικές αλλαγές θερμοκρασίας και κατακρήμνισης",
            "η κατανόηση της κλιματικής αλλαγής ενημερώνει πολιτικές για ανανεώσιμη ενέργεια και τιμολόγηση άνθρακα",
            "οι στρατηγικές προσαρμογής βοηθούν τις κοινότητες να προετοιμαστούν για πλημμύρες, ξηρασίες και καύσωνες",
        ]
    },
    {
        "name": "η νευροεπιστήμη",
        "facts": [
            "Η νευροεπιστήμη είναι η επιστημονική μελέτη του νευρικού συστήματος, συμπεριλαμβανομένου του εγκεφάλου, του νωτιαίου μυελού και των περιφερικών νεύρων.",
            "Ο ανθρώπινος εγκέφαλος περιέχει περίπου 86 δισεκατομμύρια νευρώνες, ο καθένας συνδεδεμένος με χιλιάδες άλλους.",
            "Τα δυναμικά ενέργειας είναι ηλεκτρικά σήματα που χρησιμοποιούν οι νευρώνες για να επικοινωνούν μεταξύ τους.",
            "Νευροδιαβιβαστές όπως η ντοπαμίνη, η σεροτονίνη και το γλουταμικό διαμεσολαβούν στη νευρωνική επικοινωνία.",
            "Η νευροπλαστικότητα είναι η ικανότητα του εγκεφάλου να αναδιοργανώνεται σχηματίζοντας νέες συνδέσεις.",
            "Ο ιππόκαμπος παίζει κρίσιμο ρόλο στη δημιουργία και εδραίωση μακροπρόθεσμων αναμνήσεων.",
            "Ο ύπνος είναι απαραίτητος για την εδραίωση αναμνήσεων και την αποβολή νευροτοξικών αποβλήτων από τον εγκέφαλο.",
            "Τεχνικές απεικόνισης όπως fMRI, EEG και PET επιτρέπουν στους επιστήμονες να παρατηρούν εγκεφαλική δραστηριότητα.",
            "Διαταραχές όπως το Alzheimer, η κατάθλιψη, η σχιζοφρένεια και η επιληψία αφορούν διαταραχές στη λειτουργία του εγκεφάλου.",
            "Η οπτογενετική είναι τεχνική που χρησιμοποιεί φως για τον έλεγχο συγκεκριμένων νευρώνων.",
        ],
        "applications": [
            "η νευροεπιστήμη οδηγεί στην ανάπτυξη νέων θεραπειών για νευρολογικές και ψυχιατρικές διαταραχές",
            "οι διεπαφές εγκεφάλου-υπολογιστή μπορούν να αποκαταστήσουν κινητικότητα σε παραλυμένους ασθενείς",
            "οι γνώσεις από τη νευροεπιστήμη βελτιώνουν στρατηγικές μάθησης και εκπαιδευτικές μεθοδολογίες",
        ]
    },
    {
        "name": "η ελληνική φιλοσοφία",
        "facts": [
            "Η ελληνική φιλοσοφία ξεκίνησε τον 6ο αιώνα π.Χ. με τους Προσωκρατικούς, που αναζητούσαν φυσικές εξηγήσεις για τον κόσμο.",
            "Ο Σωκράτης εισήγαγε τη διαλεκτική μέθοδο, εξετάζοντας πεποιθήσεις μέσω ερωταποκρίσεων.",
            "Ο Πλάτωνας ανέπτυξε τη θεωρία των Ιδεών, υποστηρίζοντας ότι η πραγματικότητα αποτελείται από αφηρημένες μορφές.",
            "Ο Αριστοτέλης δημιούργησε συστηματική φιλοσοφία, καλύπτοντας λογική, βιολογία, φυσική, ηθική και πολιτική.",
            "Ο Σωκράτης δεν έγραψε τίποτα· η διδασκαλία του γνωρίζουμε μέσα από τους διαλόγους του Πλάτωνα.",
            "Η Στωική φιλοσοφία δίδασκε ότι η αρετή και η λογική είναι αρκετές για την ευδαιμονία.",
            "Ο Επίκουρος υποστήριξε ότι η ηδονή και η αταραξία αποτελούν το ύψιστο αγαθό.",
            "Η αλληγορία του σπηλαίου του Πλάτωνα απεικονίζει τη διαφορά μεταξύ γνώσης και γνώμης.",
            "Η ελληνική φιλοσοφία επηρέασε βαθιά τη ρωμαϊκή, ισλαμική και δυτική ευρωπαϊκή σκέψη.",
            "Σύγχρονα πεδία όπως η λογική, η ηθική, η μεταφυσική και η πολιτική φιλοσοφία έχουν ρίζες στην ελληνική σκέψη.",
        ],
        "applications": [
            "η Σωκρατική μέθοδος χρησιμοποιείται ευρέως στην εκπαίδευση για ανάπτυξη κριτικής σκέψης",
            "η αριστοτελική λογική αποτέλεσε τη βάση για τα μαθηματικά και την επιστημονική μέθοδο",
            "η στωική φιλοσοφία βρίσκει εφαρμογή στη σύγχρονη γνωσιακή συμπεριφορική θεραπεία",
        ]
    },
    {
        "name": "η βιομηχανική επανάσταση",
        "facts": [
            "Η Βιομηχανική Επανάσταση ξεκίνησε στη Βρετανία τα τέλη του 18ου αιώνα και μεταμόρφωσε την παραγωγή, τη μεταφορά και την κοινωνία.",
            "Η ατμομηχανή, βελτιωμένη από τον James Watt, αποτέλεσε κινητήριο δύναμη της βιομηχανοποίησης.",
            "Η μαζική παραγωγή αντικατέστησε τη χειροτεχνία, μειώνοντας το κόστος και αυξάνοντας την παραγωγικότητα.",
            "Η αστικοποίηση επιταχύνθηκε καθώς εργάτες μεταναστεύσαν από την ύπαιθρο σε βιομηχανικές πόλεις.",
            "Νέα υλικά όπως ο σίδηρος και ο χάλυβας έγιναν βασικά για κατασκευές, σιδηροδρόμους και μηχανήματα.",
            "Η εργατική τάξη αναδύθηκε ως κοινωνική κατηγορία, γεννώντας το εργατικό κίνημα και συνδικάτα.",
            "Παιδική εργασία και άθλιες συνθήκες στα εργοστάσια οδήγησαν σε μεταρρυθμίσεις και εργατική νομοθεσία.",
            "Η Βιομηχανική Επανάσταση διέδωσε τελικά σε ολόκληρη την Ευρώπη, τις ΗΠΑ και πέρα από αυτές.",
            "Η κλιματική αλλαγή έχει τις ρίζες της εν μέρει στις εκπομπές CO2 που ξεκίνησαν κατά την Βιομηχανική Επανάσταση.",
            "Η τεχνολογική καινοτομία αυτής της εποχής έθεσε τις βάσεις για τη σύγχρονη παγκόσμια οικονομία.",
        ],
        "applications": [
            "η κατανόηση της Βιομηχανικής Επανάστασης εξηγεί τις ρίζες της σύγχρονης οικονομικής ανισότητας",
            "ιστορικά μοντέλα βιομηχανοποίησης ενημερώνουν τις αναπτυξιακές στρατηγικές των σημερινών χωρών",
            "η εργατική νομοθεσία που γεννήθηκε τότε αποτελεί τη βάση της σύγχρονης εργασιακής προστασίας",
        ]
    },
    {
        "name": "η γενετική",
        "facts": [
            "Η γενετική είναι ο κλάδος της βιολογίας που μελετά τα γονίδια, την κληρονομικότητα και τη γενετική ποικιλομορφία.",
            "Ο Gregor Mendel, μελετώντας μπιζέλια, έθεσε τις βάσεις της κλασικής γενετικής τον 19ο αιώνα.",
            "Τα γονίδια είναι τμήματα DNA που κωδικοποιούν οδηγίες για την παραγωγή πρωτεϊνών.",
            "Ο ανθρώπινος γονιδιωματικός χάρτης περιέχει περίπου 3 δισεκατομμύρια ζεύγη βάσεων και περίπου 20.000 γονίδια.",
            "Μεταλλάξεις στο DNA μπορούν να οδηγήσουν σε ασθένειες όπως ο καρκίνος, η κυστική ίνωση και η σικλωβολίδα.",
            "Η τεχνολογία CRISPR-Cas9 επιτρέπει ακριβή επεξεργασία γονιδίων, ανοίγοντας νέους ορίζοντες στην ιατρική.",
            "Η γενετική διαγνωστική μπορεί να εντοπίσει κληρονομικά νοσήματα πριν ακόμα εκδηλωθούν συμπτώματα.",
            "Η εξέλιξη οδηγείται από γενετικές μεταλλάξεις και φυσική επιλογή που ευνοεί προσαρμοστικά χαρακτηριστικά.",
            "Η φαρμακογενωμική χρησιμοποιεί γενετικά δεδομένα για να εξατομικεύσει τη φαρμακευτική αγωγή.",
            "Η γενετική δακτυλοσκόπηση χρησιμοποιείται στην εγκληματολογία για ταυτοποίηση ατόμων με μεγάλη ακρίβεια.",
        ],
        "applications": [
            "η γονιδιακή θεραπεία αποσκοπεί στη διόρθωση γενετικών ελαττωμάτων που προκαλούν κληρονομικές ασθένειες",
            "η γεωργική βιοτεχνολογία χρησιμοποιεί γενετική γνώση για την ανάπτυξη ανθεκτικών και αποδοτικότερων καλλιεργειών",
            "η γενετική επιδημιολογία αποκαλύπτει τους γενετικούς παράγοντες κινδύνου για κοινές ασθένειες",
        ]
    },
    {
        "name": "η δημοκρατία",
        "facts": [
            "Η δημοκρατία είναι σύστημα διακυβέρνησης όπου η εξουσία ανήκει στον λαό, που την ασκεί άμεσα ή μέσω εκλεγμένων αντιπροσώπων.",
            "Η αρχαία Αθήνα εισήγαγε την άμεση δημοκρατία τον 5ο αιώνα π.Χ., επιτρέποντας στους πολίτες να συμμετέχουν στη νομοθεσία.",
            "Οι σύγχρονες αντιπροσωπευτικές δημοκρατίες εκλέγουν ανώτατα αξιώματα μέσω ελεύθερων και δίκαιων εκλογών.",
            "Βασικές αρχές περιλαμβάνουν την ελευθερία λόγου, το κράτος δικαίου, τη διάκριση εξουσιών και την προστασία μειονοτήτων.",
            "Η Μεγάλη Χάρτα (1215) ήταν πρώιμο ορόσημο για τον περιορισμό της εξουσίας των ηγεμόνων.",
            "Η φιλελεύθερη δημοκρατία συνδυάζει δημοκρατικές εκλογές με προστασία ατομικών ελευθεριών.",
            "Η ψηφιακή τεχνολογία δημιουργεί νέες ευκαιρίες και απειλές για τη δημοκρατική συμμετοχή.",
            "Αυταρχικά καθεστώτα συχνά διεξάγουν εκλογές αλλά υπονομεύουν τις δημοκρατικές νόρμες.",
            "Η συμμετοχή των πολιτών, η ανεξάρτητη δικαιοσύνη και ο ελεύθερος Τύπος είναι ζωτικές για υγιείς δημοκρατίες.",
            "Η εξάπλωση της δημοκρατίας επιταχύνθηκε μετά τον Β' Παγκόσμιο Πόλεμο και μετά τον Ψυχρό Πόλεμο.",
        ],
        "applications": [
            "η μελέτη της δημοκρατίας βοηθά στον εντοπισμό των συνθηκών υπό τις οποίες ελεύθερες κοινωνίες ανθίζουν",
            "τα δημοκρατικά θεσμικά όργανα παρέχουν πλαίσια για ειρηνική μεταβίβαση εξουσίας",
            "η πολιτική παιδεία ριζωμένη σε δημοκρατικές αξίες προετοιμάζει τους πολίτες για συμμετοχή στη διακυβέρνηση",
        ]
    },
    {
        "name": "η εξέλιξη",
        "facts": [
            "Η εξέλιξη είναι η αλλαγή στα κληρονομικά χαρακτηριστικά πληθυσμών κατά τη διαδοχή γενεών.",
            "Ο Κάρολος Δαρβίνος πρότεινε τη θεωρία της φυσικής επιλογής στο 'Η Καταγωγή των Ειδών' το 1859.",
            "Η φυσική επιλογή ευνοεί χαρακτηριστικά που αυξάνουν την πιθανότητα επιβίωσης και αναπαραγωγής.",
            "Οι γενετικές μεταλλάξεις είναι η κύρια πηγή νέας παραλλακτικότητας στους πληθυσμούς.",
            "Η ειδογένεση συμβαίνει όταν πληθυσμοί αποκόπτονται αναπαραγωγικά και αποκλίνουν με τον χρόνο.",
            "Το απολιθωματολογικό αρχείο παρέχει άμεσες αποδείξεις εξελικτικής αλλαγής μέσα στον γεωλογικό χρόνο.",
            "Όλη η ζωή στη Γη έχει κοινό πρόγονο, όπως υποστηρίζεται από τη μοριακή βιολογία και γενετική.",
            "Η συγκλίνουσα εξέλιξη συμβαίνει όταν άσχετα είδη αναπτύσσουν ανεξάρτητα παρόμοια χαρακτηριστικά.",
            "Η ανθρώπινη εξέλιξη περιελάμβανε αρκετά είδη hominini, με τον Homo sapiens να εμφανίζεται πριν από περίπου 300.000 χρόνια.",
            "Η εξελικτική βιολογία ενημερώνει την ιατρική, τη γεωργία και την κατανόηση της βιοποικιλότητας.",
        ],
        "applications": [
            "οι εξελικτικές αρχές εξηγούν την ανάπτυξη αντοχής στα αντιβιοτικά στα βακτήρια",
            "η φυλογενετική χρησιμοποιεί εξελικτικά δεδομένα για ταξινόμηση οργανισμών",
            "η βιολογία διατήρησης εφαρμόζει εξελικτική σκέψη για την προστασία απειλούμενων ειδών",
        ]
    },
]

# ===================================================================
#  QUESTION TEMPLATES
# ===================================================================

Q_EN = [
    "Tell me about {}.", "What is {}?", "Explain {} to me.",
    "Can you describe {}?", "What do you know about {}?",
    "I'm curious about {}.", "What are the basics of {}?",
    "How does {} work?", "Why is {} important?",
    "What makes {} significant?", "Give me an overview of {}.",
    "What should I know about {}?", "Can you break down {} for me?",
]

Q_GR = [
    "Πες μου για {}.", "Τι είναι {};", "Εξήγησε μου {}.",
    "Μπορείς να περιγράψεις {};", "Τι γνωρίζεις για {};",
    "Είμαι περίεργος/η για {}.", "Ανάλυσε {} για μένα.",
    "Πώς λειτουργεί το {};", "Γιατί είναι σημαντικό το {};",
    "Δώσε μου μια επισκόπηση του {}.", "Εξήγησε τα βασικά του {}.",
]

FOLLOW_Q_EN = [
    "Why is this important?", "How does this affect everyday life?",
    "What are the key takeaways?", "What are the practical applications?",
    "Is there anything surprising about this?",
    "What does this mean for the future?",
    "How was this discovered or developed?",
    "What challenges remain in this field?",
]

FOLLOW_Q_GR = [
    "Γιατί είναι σημαντικό;", "Πώς επηρεάζει την καθημερινή ζωή;",
    "Ποια είναι τα κυριότερα συμπεράσματα;",
    "Ποιες είναι οι πρακτικές εφαρμογές;",
    "Υπάρχει κάτι εκπληκτικό σχετικά με αυτό;",
    "Τι σημαίνει αυτό για το μέλλον;",
    "Πώς ανακαλύφθηκε ή αναπτύχθηκε αυτό;",
]

EMOTION_TAGS = ["neutral", "neutral", "neutral", "curiosity", "encouragement"]

GREET_EN = ["Hello!", "Hi!", "Hey there!", "Hi Aether!"]
GREET_GR = ["Γεια!", "Γεια σου!", "Χαίρε!", "Γεια σου Aether!"]
GREET_A_EN = "Hello! I'm Aether. I'm happy to help you learn."
GREET_A_GR = "Γεια! Είμαι ο Aether. Χαίρομαι να σε βοηθήσω να μάθεις."

BYE_EN = ["Thanks, that's all!", "Goodbye!", "That was helpful, thanks!"]
BYE_GR = ["Ευχαριστώ, αυτά ήθελα!", "Αντίο!", "Πολύ χρήσιμο, ευχαριστώ!"]
BYE_A_EN = "You're welcome! Feel free to ask anytime."
BYE_A_GR = "Παρακαλώ! Ρώτα με ό,τι θέλεις."

# ===================================================================
#  TEXT BUILDERS — coherent, fact-based
# ===================================================================

def build_en_paragraph(subj: dict) -> str:
    """Build a coherent English paragraph using real subject facts."""
    facts = random.sample(subj["facts"], random.randint(2, 4))
    app = pick(subj["applications"])
    name = subj["name"]

    openers = [
        f"Studying {name} reveals deep truths about how our world works.",
        f"Let's explore what {name} really means and why it matters.",
        f"The topic of {name} is one of the most important in modern science.",
        f"Few subjects are as rich and illuminating as {name}.",
    ]

    connectors = [
        "Furthermore,", "In addition,", "Moreover,", "Building on this,",
        "Crucially,", "Notably,", "For context,", "As an example,",
        "A key point is that", "What is particularly striking is that",
        "In fact,", "Specifically,", "This is because",
        "Another important dimension is that",
    ]

    closers = [
        f"All of this helps explain why {app}.",
        f"This is why {app}, with far-reaching implications for science and society.",
        f"In practical terms, {app}, shaping how we understand the world.",
        f"As research advances, {app}, opening new doors we are only beginning to explore.",
        f"Understanding {name} is ultimately about recognizing that {app}.",
        facts[-1],
    ]

    start_with_fact = random.random() < 0.45
    return _natural_join(facts, connectors, closers, name, app, start_with_fact)


def build_gr_paragraph(subj: dict) -> str:
    """Build a coherent Greek paragraph using real subject facts."""
    facts = random.sample(subj["facts"], random.randint(2, 4))
    app = pick(subj["applications"])
    name = subj["name"]

    openers = [
        f"{name.capitalize()} είναι ένα από τα πιο σημαντικά πεδία στη σύγχρονη επιστήμη και σκέψη.",
        f"Ας εξερευνήσουμε τι σημαίνει πραγματικά {name} και γιατί έχει σημασία.",
        f"Η εξερεύνηση {name} αποκαλύπτει θεμελιώδεις αλήθειες για τον κόσμο μας.",
        f"Λίγα θέματα είναι τόσο πλούσια και αποκαλυπτικά όσο {name}.",
    ]

    connectors = [
        "Επιπλέον,", "Επίσης,", "Ακόμα,", "Πέρα από αυτό,",
        "Σημαντικό είναι και ότι", "Αξίζει επίσης να σημειωθεί ότι",
        "Για παράδειγμα,", "Συγκεκριμένα,", "Αξιοσημείωτο είναι ότι",
        "Μια βασική διάσταση είναι ότι", "Αυτό συμβαίνει γιατί",
    ]

    closers = [
        f"Όλα αυτά βοηθούν να εξηγήσουμε γιατί {app}.",
        f"Γι' αυτό {app}, με σημαντικές επιπτώσεις για την επιστήμη και την κοινωνία.",
        f"Στην πράξη, {app}, διαμορφώνοντας τον τρόπο που βλέπουμε τον κόσμο.",
        f"Καθώς η έρευνα προχωρά, {app}, ανοίγοντας νέους ορίζοντες.",
        f"Η κατανόηση {name} σημαίνει τελικά ότι {app}.",
        facts[-1],
    ]

    start_with_fact = random.random() < 0.45
    return _natural_join(facts, connectors, closers, name, app, start_with_fact)


def build_en_answer(subj: dict) -> str:
    """Build a coherent English Q&A answer — conversational, not like a list."""
    facts = random.sample(subj["facts"], random.randint(3, 5))
    app = pick(subj["applications"])
    name = subj["name"]

    intro_options = [
        f"That is an excellent question about {name}.",
        f"I am glad you asked about {name}. It connects to so many interesting fields.",
        f"Great question! Let me break down {name} in a way that makes sense.",
        f"Let me tell you about {name}.",
    ]

    connectors_mid = [
        "To start with,", "A good place to begin is that",
        "One important thing to understand is that",
        "The first key point is that",
    ]
    connectors_more = [
        "Beyond that,", "Another crucial point is that",
        "Furthermore,", "In addition to this,",
        "It is also important to know that",
        "What is really interesting is that",
        "Here is another aspect:",
        "Adding to that,", "Moreover,", "On top of that,",
    ]

    closers = [
        f"When we look at the real-world impact, {app}. That is what makes {name} so important.",
        f"This has direct practical significance: {app}. That is why studying {name} matters so much.",
        f"The takeaway is that {app}. Understanding {name} helps us see the bigger picture.",
        f"In practice, {app}. This is yet another reason why this topic stays so relevant.",
    ]

    parts = [pick(intro_options)]
    parts.append(_smart_connector(pick(connectors_mid), facts[0]))
    for fact in facts[1:]:
        if random.random() < 0.3:
            parts.append(pick(closers))
            break
        parts.append(_smart_connector(pick(connectors_more), fact))
    else:
        parts.append(pick(closers))

    return " ".join(parts)


def build_gr_answer(subj: dict) -> str:
    """Build a coherent Greek Q&A answer — conversational, not like a list."""
    facts = random.sample(subj["facts"], random.randint(3, 5))
    app = pick(subj["applications"])
    name = subj["name"]

    intro_options = [
        f"Πολύ καλή ερώτηση για {name}.",
        f"Χαίρομαι που ρωτάς για {name}. Συνδέεται με τόσα πολλά ενδιαφέροντα πεδία.",
        f"Εξαιρετική ερώτηση! Ας δούμε {name} με τρόπο που να βγάζει νόημα.",
        f"Ας σου μιλήσω για {name}.",
    ]

    connectors_mid = [
        "Για να ξεκινήσουμε,", "Ένα καλό σημείο εκκίνησης είναι ότι",
        "Ένα σημαντικό πράγμα που πρέπει να κατανοήσεις είναι ότι",
        "Το βασικό σημείο είναι ότι",
    ]
    connectors_more = [
        "Πέρα από αυτό,", "Ένα ακόμα κρίσιμο σημείο είναι ότι",
        "Επιπλέον,", "Εκτός από αυτό,",
        "Είναι επίσης σημαντικό να γνωρίζεις ότι",
        "Αυτό που είναι πραγματικά ενδιαφέρον είναι ότι",
        "Ακόμα ένα στοιχείο:", "Επιπρόσθετα,",
    ]

    closers = [
        f"Σε πρακτικό επίπεδο, {app}. Γι' αυτό {name} έχει τόση αξία.",
        f"Αυτό έχει άμεση πρακτική σημασία: {app}. Γι' αυτό αξίζει την προσοχή μας.",
        f"Το συμπέρασμα είναι ότι {app}. Έτσι κατανοούμε καλύτερα τη μεγάλη εικόνα.",
        f"Στην πράξη, {app}. Αυτός είναι ένας λόγος που {name} αξίζει να μελετάται.",
    ]

    parts = [pick(intro_options)]
    parts.append(_smart_connector(pick(connectors_mid), facts[0]))
    for fact in facts[1:]:
        if random.random() < 0.3:
            parts.append(pick(closers))
            break
        parts.append(_smart_connector(pick(connectors_more), fact))
    else:
        parts.append(pick(closers))

    return " ".join(parts)


def build_follow_answer_en(subj: dict) -> str:
    name = subj['name']
    app = pick(subj["applications"])
    options = [
        f"That is a great follow-up. The real significance of {name} becomes clear when we see how {app}. "
        f"This connects to so many other fields, which is what makes this topic so central.",
        f"Great question. The importance of {name} is that {app}. "
        f"As we learn more, the links to other areas of knowledge only grow stronger.",
        f"That is precisely the point. {app.capitalize()}. "
        f"It is not just theoretical — it has real, tangible effects on our lives.",
    ]
    return pick(options)


def build_follow_answer_gr(subj: dict) -> str:
    name = subj['name']
    app = pick(subj["applications"])
    options = [
        f"Καλή συνέχεια. Η πραγματική σημασία του {name} γίνεται σαφής όταν βλέπουμε πώς {app}. "
        f"Αυτό συνδέεται με τόσα άλλα πεδία, γι' αυτό το θέμα είναι τόσο κεντρικό.",
        f"Καλή ερώτηση. Η σημασία του {name} είναι ότι {app}. "
        f"Καθώς μαθαίνουμε περισσότερα, οι συνδέσεις με άλλους τομείς γνώσης δυναμώνουν.",
        f"Γι' αυτό ακριβώς. {app.capitalize()}. "
        f"Δεν είναι μόνο θεωρητικό — έχει πραγματική, χειροπιαστή επίδραση στη ζωή μας.",
    ]
    return pick(options)


# ===================================================================
#  ENTRY BUILDERS
# ===================================================================

def raw_entry(text: str) -> dict:
    return {"text": text}


def qa_entry(q: str, a: str, emo: str) -> dict:
    return {"text": f"User: {q}\n\nAether: <{emo}> {a}"}


def multi_entry(turns: list) -> dict:
    parts = []
    for u, a, emo in turns:
        parts.append(f"User: {u}\n\nAether: <{emo}> {a}")
    return {"text": "\n\n".join(parts)}


# ===================================================================
#  DATASET BALANCE HELPERS
# ===================================================================

TARGET_BYTES = 512 * 1024 * 1024  # 512 MiB output file
RAW_RATIO = 0.40


def entry_bytes(entry):
    return len(json.dumps(entry, ensure_ascii=False).encode("utf-8")) + 1


def take_until_bytes(entries, byte_limit):
    kept, total = [], 0
    for e in entries:
        sz = entry_bytes(e)
        if total + sz > byte_limit:
            break
        kept.append(e)
        total += sz
    return kept, total


def interleave_entries(raw_list, qa_list, raw_per=2, qa_per=3):
    """Ordered interleave: 2 raw, 3 Q&A, repeat — no random shuffle."""
    out, ri, qi = [], 0, 0
    while ri < len(raw_list) or qi < len(qa_list):
        for _ in range(raw_per):
            if ri < len(raw_list):
                out.append(raw_list[ri])
                ri += 1
        for _ in range(qa_per):
            if qi < len(qa_list):
                out.append(qa_list[qi])
                qi += 1
    return out


def fill_raw_to_target(raw_kept, raw_bytes, raw_target):
    lang_toggle = 0
    while raw_bytes < raw_target:
        if lang_toggle % 2 == 0:
            subj = pick_subject(SUBJECTS_EN, identity_weight=0.02)
            entry = raw_entry(build_en_paragraph(subj))
        else:
            subj = pick_subject(SUBJECTS_GR, identity_weight=0.02)
            entry = raw_entry(build_gr_paragraph(subj))
        lang_toggle += 1
        sz = entry_bytes(entry)
        if raw_bytes + sz > raw_target:
            break
        raw_kept.append(entry)
        raw_bytes += sz
    return raw_kept, raw_bytes


def fill_qa_to_target(qa_kept, qa_bytes, qa_target):
    lang_toggle = 0
    while qa_bytes < qa_target:
        if lang_toggle % 2 == 0:
            subj = pick_subject(SUBJECTS_EN, identity_weight=0.02)
            entry = qa_entry(pick(Q_EN).format(subj["name"]), build_en_answer(subj), pick(EMOTION_TAGS))
        else:
            subj = pick_subject(SUBJECTS_GR, identity_weight=0.02)
            entry = qa_entry(pick(Q_GR).format(subj["name"]), build_gr_answer(subj), pick(EMOTION_TAGS))
        lang_toggle += 1
        sz = entry_bytes(entry)
        if qa_bytes + sz > qa_target:
            break
        qa_kept.append(entry)
        qa_bytes += sz
    return qa_kept, qa_bytes


# ===================================================================
#  MAIN GENERATION
# ===================================================================

def generate():
    target_bytes = TARGET_BYTES
    output_path = "aether_dataset.jsonl"
    total = 0

    with open(output_path, "w", encoding="utf-8") as f:

        def write(entry):
            nonlocal total
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1

        # ── Phase 1: EN raw paragraphs ──
        print("Phase 1: EN raw paragraphs (pre-training style)...")
        for pass_num in range(3500):
            for idx, subj in enumerate(SUBJECTS_EN):
                if idx == 0 and random.random() > 0.05:
                    continue
                write(raw_entry(build_en_paragraph(subj)))

        # ── Phase 2: GR raw paragraphs ──
        print("Phase 2: GR raw paragraphs (pre-training style)...")
        for pass_num in range(3500):
            for idx, subj in enumerate(SUBJECTS_GR):
                if idx == 0 and random.random() > 0.05:
                    continue
                write(raw_entry(build_gr_paragraph(subj)))

        # ── Phase 3: EN single-turn Q&A ──
        print("Phase 3: EN Q&A (User/Aether format)...")
        for pass_num in range(300):
            for idx, subj in enumerate(SUBJECTS_EN):
                if idx == 0 and random.random() > 0.05:
                    continue
                q = pick(Q_EN).format(subj["name"])
                a = build_en_answer(subj)
                emo = pick(EMOTION_TAGS)
                write(qa_entry(q, a, emo))

        # ── Phase 4: GR single-turn Q&A ──
        print("Phase 4: GR Q&A (User/Aether format)...")
        for pass_num in range(300):
            for idx, subj in enumerate(SUBJECTS_GR):
                if idx == 0 and random.random() > 0.05:
                    continue
                q = pick(Q_GR).format(subj["name"])
                a = build_gr_answer(subj)
                emo = pick(EMOTION_TAGS)
                write(qa_entry(q, a, emo))

        # ── Phase 5: EN multi-turn conversations ──
        print("Phase 5: EN multi-turn (User/Aether format)...")
        for pass_num in range(200):
            for idx, subj in enumerate(SUBJECTS_EN):
                if idx == 0 and random.random() > 0.05:
                    continue
                q = pick(Q_EN).format(subj["name"])
                a = build_en_answer(subj)
                fq = pick(FOLLOW_Q_EN)
                fa = build_follow_answer_en(subj)
                turns = [
                    (pick(GREET_EN), GREET_A_EN, "joy"),
                    (q, a, pick(EMOTION_TAGS)),
                    (fq, fa, "neutral"),
                    (pick(BYE_EN), BYE_A_EN, "joy"),
                ]
                write(multi_entry(turns))

        # ── Phase 6: GR multi-turn conversations ──
        print("Phase 6: GR multi-turn (User/Aether format)...")
        for pass_num in range(200):
            for idx, subj in enumerate(SUBJECTS_GR):
                if idx == 0 and random.random() > 0.05:
                    continue
                q = pick(Q_GR).format(subj["name"])
                a = build_gr_answer(subj)
                fq = pick(FOLLOW_Q_GR)
                fa = build_follow_answer_gr(subj)
                turns = [
                    (pick(GREET_GR), GREET_A_GR, "joy"),
                    (q, a, pick(EMOTION_TAGS)),
                    (fq, fa, "neutral"),
                    (pick(BYE_GR), BYE_A_GR, "joy"),
                ]
                write(multi_entry(turns))

    pre_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nPhases 1-6 done: {total:,} entries, {pre_mb:.1f} MB (temporary file)")

    # ── Phase 7: Balance to exactly 512MB — 40% raw / 60% Q&A, fill if short ──
    print(f"\nPhase 7: Balancing to {target_bytes / 1024 / 1024:.0f} MB (40% raw / 60% Q&A)...")
    entries = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    raw_entries = [e for e in entries if "User:" not in e.get("text", "")]
    qa_entries = [e for e in entries if "User:" in e.get("text", "")]

    target_raw_bytes = int(target_bytes * RAW_RATIO)
    target_qa_bytes = target_bytes - target_raw_bytes

    raw_kept, raw_bytes = take_until_bytes(raw_entries, target_raw_bytes)
    qa_kept, qa_bytes = take_until_bytes(qa_entries, target_qa_bytes)

    if raw_bytes < target_raw_bytes:
        need_mb = (target_raw_bytes - raw_bytes) / 1024 / 1024
        print(f"  Generating +{need_mb:.1f} MB raw (had only {raw_bytes/1024/1024:.1f} MB)...")
        raw_kept, raw_bytes = fill_raw_to_target(raw_kept, raw_bytes, target_raw_bytes)

    if qa_bytes < target_qa_bytes:
        need_mb = (target_qa_bytes - qa_bytes) / 1024 / 1024
        print(f"  Generating +{need_mb:.1f} MB Q&A (had only {qa_bytes/1024/1024:.1f} MB)...")
        qa_kept, qa_bytes = fill_qa_to_target(qa_kept, qa_bytes, target_qa_bytes)

    balanced = interleave_entries(raw_kept, qa_kept, raw_per=2, qa_per=3)
    with open(output_path, "w", encoding="utf-8") as f:
        for e in balanced:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    final_bytes = os.path.getsize(output_path)
    mb = final_bytes / 1024 / 1024
    print(f"\n✅ Done!")
    print(f"   Total entries : {len(balanced):,} (raw {len(raw_kept):,} / Q&A {len(qa_kept):,})")
    print(f"   Raw section   : {raw_bytes/1024/1024:.1f} MB (target {target_raw_bytes/1024/1024:.1f} MB)")
    print(f"   Q&A section   : {qa_bytes/1024/1024:.1f} MB (target {target_qa_bytes/1024/1024:.1f} MB)")
    print(f"   File size     : {mb:.2f} MB ({final_bytes:,} bytes)")
    print(f"   Order         : interleaved 2 raw : 3 Q&A")


if __name__ == "__main__":
    generate()
