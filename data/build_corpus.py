"""
Build training corpus for Cortex AGI.
Combines 100K English words with comprehensive language rules.
Output: training_corpus.txt — the complete training data.
"""

import os
import random

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORDS_FILE = os.path.join(SCRIPT_DIR, "words_raw.txt")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "training_corpus.txt")


def load_words(path, max_words=100000):
    """Load and clean word list."""
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # Only pure ASCII English words
            if line.isascii() and line.isalpha():
                words.append(line.lower())
    # Deduplicate preserving order
    seen = set()
    unique = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:max_words]


def build_word_sections(words):
    """Build word training data organized by length, frequency, and letter."""
    sections = []

    # Section: All words listed (exposure)
    sections.append("SECTION: ENGLISH VOCABULARY\n" + "=" * 40 + "\n")
    for i in range(0, len(words), 20):
        batch = words[i:i+20]
        sections.append(" ".join(batch) + ".\n")

    # Section: Words by first letter
    sections.append("\nSECTION: WORDS BY LETTER\n" + "=" * 40 + "\n")
    for letter in "abcdefghijklmnopqrstuvwxyz":
        letter_words = [w for w in words[:10000] if w.startswith(letter)][:50]
        if letter_words:
            sections.append(f"\nWords starting with {letter.upper()}: {', '.join(letter_words)}.\n")

    # Section: Words by length
    sections.append("\nSECTION: WORDS BY LENGTH\n" + "=" * 40 + "\n")
    for length in range(1, 16):
        length_words = [w for w in words[:20000] if len(w) == length][:30]
        if length_words:
            sections.append(f"\n{length}-letter words: {', '.join(length_words)}.\n")

    # Section: Common word pairs and collocations
    sections.append("\nSECTION: COMMON WORD PAIRS\n" + "=" * 40 + "\n")
    pairs = [
        "the world", "of the", "in the", "to the", "and the",
        "a good", "a great", "a new", "a long", "a small",
        "very good", "very much", "very well", "very long", "very large",
        "do not", "can not", "will not", "should not", "could not",
        "has been", "have been", "had been", "will be", "would be",
        "each other", "one another", "as well", "so much", "too much",
        "right now", "just now", "even though", "as if", "such as",
        "more than", "less than", "rather than", "other than", "better than",
    ]
    for p in pairs:
        sections.append(f"{p}. {p}. {p}.\n")

    return "".join(sections)


def build_grammar_rules():
    """Comprehensive English grammar rules as training data."""
    rules = """
SECTION: PARTS OF SPEECH
========================

A noun is a word that names a person, place, thing, or idea.
Examples of nouns: cat, dog, house, city, love, freedom, teacher, mountain, book, water.
A noun can be the subject of a sentence. A noun can be the object of a sentence.
Common nouns are general: city, river, country. Proper nouns are specific: London, Nile, France.
Abstract nouns name ideas: love, justice, courage, freedom, happiness, truth, knowledge, wisdom.
Collective nouns name groups: team, family, flock, herd, class, audience, committee, crowd.

A verb is a word that shows action or state of being.
Examples of action verbs: run, jump, eat, write, think, speak, build, create, learn, teach.
Examples of state verbs: is, am, are, was, were, be, been, being, seem, appear, exist.
Linking verbs connect the subject to a description: is, am, are, was, were, seem, become, appear.
Helping verbs assist main verbs: can, could, will, would, shall, should, may, might, must, do, does, did, have, has, had.

An adjective is a word that describes a noun.
Examples of adjectives: big, small, tall, short, red, blue, fast, slow, happy, sad, beautiful, ugly.
Adjectives answer: What kind? Which one? How many? How much?
Order of adjectives: opinion, size, age, shape, color, origin, material, purpose.
A big old round red Italian leather racing car. A beautiful small new square blue American wooden toy box.

An adverb is a word that describes a verb, adjective, or another adverb.
Examples of adverbs: quickly, slowly, very, really, often, never, always, sometimes, here, there.
Many adverbs end in -ly: quickly, slowly, carefully, happily, sadly, beautifully, easily, hardly.
Adverbs of frequency: always, usually, often, sometimes, rarely, seldom, never.
Adverbs of manner: quickly, slowly, carefully, easily, well, badly, hard, fast.
Adverbs of place: here, there, everywhere, nowhere, somewhere, inside, outside, above, below.
Adverbs of time: now, then, today, yesterday, tomorrow, soon, later, already, still, yet.

A pronoun is a word that takes the place of a noun.
Subject pronouns: I, you, he, she, it, we, they.
Object pronouns: me, you, him, her, it, us, them.
Possessive pronouns: my, mine, your, yours, his, her, hers, its, our, ours, their, theirs.
Reflexive pronouns: myself, yourself, himself, herself, itself, ourselves, yourselves, themselves.
Demonstrative pronouns: this, that, these, those.
Interrogative pronouns: who, whom, whose, which, what.
Relative pronouns: who, whom, whose, which, that.

A preposition is a word that shows the relationship between a noun and another word.
Common prepositions: in, on, at, to, from, with, by, for, about, of, between, among, through, during, before, after, above, below, under, over, near, far.
Prepositional phrases: in the house, on the table, at the park, to the store, from the city, with a friend, by the river, for the team.

A conjunction is a word that connects words, phrases, or clauses.
Coordinating conjunctions: for, and, nor, but, or, yet, so. Remember: FANBOYS.
Subordinating conjunctions: because, since, although, though, while, when, if, unless, until, after, before, as, whereas.
Correlative conjunctions: both...and, either...or, neither...nor, not only...but also, whether...or.

An interjection is a word that expresses emotion.
Examples of interjections: oh, wow, ouch, hey, well, hmm, ah, oops, hurray, alas.

SECTION: SENTENCE STRUCTURE
============================

A sentence has a subject and a predicate.
The subject tells who or what the sentence is about.
The predicate tells what the subject does or is.

The cat sleeps. Subject: the cat. Predicate: sleeps.
The big dog ran quickly. Subject: the big dog. Predicate: ran quickly.
She is happy. Subject: she. Predicate: is happy.

Simple sentence: one independent clause. The sun shines.
Compound sentence: two independent clauses joined by a conjunction. The sun shines and the birds sing.
Complex sentence: one independent clause and one or more dependent clauses. When the sun shines, the birds sing.
Compound-complex sentence: two or more independent clauses and one or more dependent clauses. When the sun shines, the birds sing and the flowers bloom.

Word order in English is Subject-Verb-Object (SVO).
I eat food. She reads books. They play games. We learn languages. He writes code.
I eat food. She reads books. They play games. We learn languages. He writes code.

Questions reverse the subject and auxiliary verb.
She is happy. Is she happy?
They are coming. Are they coming?
He can swim. Can he swim?
You have finished. Have you finished?

Negative sentences use not after the auxiliary verb.
She is not happy. They are not coming. He cannot swim. I do not know. We will not go.

SECTION: VERB TENSES
=====================

Present simple: I walk. You walk. He walks. She walks. It walks. We walk. They walk.
Present continuous: I am walking. You are walking. He is walking. She is walking. We are walking. They are walking.
Present perfect: I have walked. You have walked. He has walked. She has walked. We have walked. They have walked.
Present perfect continuous: I have been walking. You have been walking. He has been walking.

Past simple: I walked. You walked. He walked. She walked. It walked. We walked. They walked.
Past continuous: I was walking. You were walking. He was walking. She was walking. We were walking. They were walking.
Past perfect: I had walked. You had walked. He had walked. She had walked. We had walked. They had walked.
Past perfect continuous: I had been walking. You had been walking. He had been walking.

Future simple: I will walk. You will walk. He will walk. She will walk. We will walk. They will walk.
Future continuous: I will be walking. You will be walking. He will be walking.
Future perfect: I will have walked. You will have walked. He will have walked.
Future perfect continuous: I will have been walking. You will have been walking.

Irregular verbs do not follow the regular -ed pattern.
be, was/were, been. have, had, had. do, did, done. go, went, gone.
see, saw, seen. come, came, come. take, took, taken. give, gave, given.
make, made, made. know, knew, known. think, thought, thought. say, said, said.
get, got, gotten. find, found, found. tell, told, told. become, became, become.
leave, left, left. feel, felt, felt. put, put, put. bring, brought, brought.
begin, began, begun. keep, kept, kept. hold, held, held. write, wrote, written.
stand, stood, stood. hear, heard, heard. let, let, let. mean, meant, meant.
set, set, set. meet, met, met. run, ran, run. pay, paid, paid.
sit, sat, sat. speak, spoke, spoken. lie, lay, lain. lead, led, led.
read, read, read. grow, grew, grown. lose, lost, lost. fall, fell, fallen.
send, sent, sent. build, built, built. understand, understood, understood.
draw, drew, drawn. break, broke, broken. spend, spent, spent.
cut, cut, cut. rise, rose, risen. drive, drove, driven. buy, bought, bought.
wear, wore, worn. choose, chose, chosen.

SECTION: PUNCTUATION RULES
============================

A period ends a declarative sentence. The cat is sleeping.
A question mark ends a question. Is the cat sleeping?
An exclamation mark ends an exclamation. The cat is so cute!
A comma separates items in a list. I bought apples, oranges, and bananas.
A comma separates clauses. When it rains, I stay inside.
A comma follows introductory words. However, the plan failed. Therefore, we tried again.
An apostrophe shows possession. The cat's toy. The dogs' bones. James's book.
An apostrophe shows contraction. I'm means I am. Don't means do not. Can't means cannot. It's means it is. They're means they are. We've means we have.
A colon introduces a list or explanation. I need three things: food, water, and shelter.
A semicolon connects related independent clauses. The sun set; the stars appeared.
Quotation marks surround spoken words. She said, "Hello." He replied, "Hi there."
Parentheses add extra information. The cat (a tabby) slept on the mat.

SECTION: WORD FORMATION
========================

Prefixes are added to the beginning of a word to change its meaning.
un- means not: unhappy, unable, unclear, unfair, unknown, unusual, unlikely, unaware.
re- means again: redo, rewrite, rebuild, rethink, reopen, restart, return, review.
pre- means before: preview, predict, prepare, prevent, preschool, prehistoric, premature.
dis- means not or opposite: disagree, disappear, disconnect, dislike, disorder, disable.
mis- means wrong: mistake, misunderstand, mislead, misplace, misspell, misjudge.
over- means too much: overdo, overeat, overflow, overlook, overcome, overwork.
under- means too little: underdo, underestimate, underground, understand, undervalue.
inter- means between: international, interact, internet, interview, interrupt, interval.
super- means above: superhero, supernatural, supermarket, supervise, superior.
anti- means against: antiwar, antibody, antifreeze, antisocial, antivirus.

Suffixes are added to the end of a word to change its meaning or part of speech.
-er makes a doer: teacher, worker, player, singer, writer, builder, reader, leader.
-est makes superlative: biggest, smallest, tallest, fastest, slowest, longest.
-tion or -sion makes a noun: action, education, information, decision, discussion, expression.
-ment makes a noun: movement, government, development, management, agreement.
-ness makes a noun: happiness, sadness, darkness, kindness, weakness, illness.
-able or -ible means can be: readable, breakable, visible, possible, flexible, edible.
-ful means full of: beautiful, wonderful, powerful, hopeful, careful, grateful.
-less means without: hopeless, careless, fearless, endless, homeless, useless.
-ly makes an adverb: quickly, slowly, carefully, happily, sadly, easily, rarely.
-ous means full of: dangerous, famous, nervous, curious, generous, mysterious.

SECTION: SPELLING RULES
========================

I before E except after C: believe, receive, achieve, deceive, perceive, conceive.
Exceptions: weird, seize, neither, their, height, foreign, science.

Drop the silent E before adding a suffix starting with a vowel: make becomes making, write becomes writing, hope becomes hoping, come becomes coming.
Keep the E before a suffix starting with a consonant: hopeful, movement, careful, lonely.

Double the final consonant when a one-syllable word ends in consonant-vowel-consonant: run becomes running, sit becomes sitting, stop becomes stopping, big becomes bigger, hot becomes hotter.

Change Y to I before adding a suffix: happy becomes happiness, carry becomes carried, easy becomes easier, beauty becomes beautiful.
But keep Y before -ing: carrying, studying, playing, saying.

Add -es to words ending in s, sh, ch, x, z: buses, dishes, watches, boxes, buzzes.
Add -s to most other words: cats, dogs, books, cars, trees.
Change f or fe to ves: life becomes lives, knife becomes knives, wolf becomes wolves, leaf becomes leaves.

SECTION: SENTENCE PATTERNS
============================

Subject + Verb: Birds fly. Fish swim. Dogs bark. Cats meow. Rain falls. Wind blows.

Subject + Verb + Object: I read books. She writes letters. They play music. We eat food. He drinks water.

Subject + Verb + Adjective: She is tall. He seems happy. They look tired. It feels cold. We are ready.

Subject + Verb + Adverb: She runs quickly. He speaks softly. They work hard. We arrived early.

Subject + Verb + Indirect Object + Direct Object: She gave him a book. He told her a story. They sent us a letter. I showed them the way.

There is and There are: There is a cat on the mat. There are three books on the shelf. There is water in the glass. There are many stars in the sky.

SECTION: ARTICLES AND DETERMINERS
==================================

A is used before consonant sounds: a cat, a dog, a book, a house, a university.
An is used before vowel sounds: an apple, an egg, an idea, an hour, an umbrella.
The is used for specific things: the sun, the moon, the earth, the sky, the ocean.
The is used when both speaker and listener know which one: the car, the door, the answer.
No article for general plurals: Cats are animals. Dogs are loyal. Books are important.
No article for uncountable nouns in general: Water is essential. Knowledge is power. Music is beautiful.

SECTION: COMPARATIVE AND SUPERLATIVE
=====================================

One-syllable adjectives: add -er and -est.
tall, taller, tallest. short, shorter, shortest. fast, faster, fastest. slow, slower, slowest.
big, bigger, biggest. small, smaller, smallest. old, older, oldest. young, younger, youngest.

Two or more syllables: use more and most.
beautiful, more beautiful, most beautiful. interesting, more interesting, most interesting.
important, more important, most important. difficult, more difficult, most difficult.

Irregular comparatives: good, better, best. bad, worse, worst. far, farther, farthest. little, less, least. much, more, most. many, more, most.

SECTION: CONDITIONAL SENTENCES
================================

Zero conditional: If you heat water to 100 degrees, it boils. If you mix red and blue, you get purple.
First conditional: If it rains tomorrow, I will stay home. If she studies hard, she will pass the exam.
Second conditional: If I were rich, I would travel the world. If he had time, he would learn to play guitar.
Third conditional: If I had studied harder, I would have passed. If they had left earlier, they would have arrived on time.

SECTION: PASSIVE VOICE
========================

Active: The cat caught the mouse. Passive: The mouse was caught by the cat.
Active: She writes the letter. Passive: The letter is written by her.
Active: They built the house. Passive: The house was built by them.
Active: He will finish the work. Passive: The work will be finished by him.
Active: We have completed the project. Passive: The project has been completed by us.

SECTION: REPORTED SPEECH
==========================

Direct: She said, "I am happy." Reported: She said that she was happy.
Direct: He said, "I will come." Reported: He said that he would come.
Direct: They said, "We are leaving." Reported: They said that they were leaving.
Direct: She asked, "Where is the station?" Reported: She asked where the station was.
Direct: He said, "I can help." Reported: He said that he could help.

SECTION: MODAL VERBS
=====================

Can expresses ability: I can swim. She can speak French. They can solve problems.
Could expresses past ability or possibility: I could swim when I was five. It could rain tomorrow.
Will expresses future or willingness: I will help you. The sun will rise tomorrow. She will be there.
Would expresses conditional or polite request: I would go if I could. Would you please help me?
Shall expresses suggestion or offer: Shall we go? Shall I help you? We shall overcome.
Should expresses advice or obligation: You should study. She should see a doctor. They should be careful.
May expresses permission or possibility: May I come in? It may rain later. She may be right.
Might expresses possibility: It might snow tonight. He might come to the party. They might agree.
Must expresses necessity or strong obligation: You must stop. I must go now. She must be tired.

SECTION: COMMON ENGLISH IDIOMS
================================

Break the ice means to start a conversation. She broke the ice with a joke.
Hit the nail on the head means to be exactly right. You hit the nail on the head.
Piece of cake means something easy. The test was a piece of cake.
Under the weather means feeling sick. I am under the weather today.
Cost an arm and a leg means very expensive. That car costs an arm and a leg.
Let the cat out of the bag means to reveal a secret. She let the cat out of the bag.
A blessing in disguise means something good that seemed bad at first.
Better late than never means it is better to do something late than not at all.
Bite the bullet means to endure something difficult. We had to bite the bullet and start over.
Break a leg means good luck. Break a leg at your performance tonight.
Call it a day means to stop working. Let us call it a day.
Cut to the chase means to get to the point. Cut to the chase and tell me what happened.
Easy come, easy go means something gained easily is lost easily.
Get out of hand means to become uncontrollable. The situation got out of hand.
Hang in there means to persevere. Hang in there, things will get better.
It takes two to tango means both parties are involved. It takes two to tango.
Kill two birds with one stone means to accomplish two things with one action.
Once in a blue moon means very rarely. I only see him once in a blue moon.
Speak of the devil means the person you were talking about just appeared.
The best of both worlds means to have advantages of two different things.
Time flies means time passes quickly. Time flies when you are having fun.

SECTION: QUESTION WORDS
=========================

Who asks about a person. Who is that? Who wrote this? Who are you? Who won the game?
What asks about a thing or idea. What is this? What happened? What time is it? What do you want?
Where asks about a place. Where is the store? Where do you live? Where are we going?
When asks about time. When is the meeting? When did you arrive? When will it start?
Why asks about a reason. Why are you late? Why did she leave? Why is the sky blue?
How asks about manner or degree. How are you? How did you do it? How far is it? How much does it cost?
Which asks about a choice. Which one do you want? Which color do you prefer? Which way should we go?
Whose asks about possession. Whose book is this? Whose idea was it? Whose turn is it?

SECTION: COMMON SENTENCE TEMPLATES
====================================

The [noun] is [adjective]. The sky is blue. The cat is small. The mountain is tall. The water is cold.
The [noun] [verb] the [noun]. The cat chased the mouse. The dog ate the bone. The wind moved the leaves.
[Pronoun] [verb] [adverb]. She spoke softly. He ran quickly. They arrived early. We worked hard.
If [clause], then [clause]. If it rains, then we stay inside. If you study, then you learn.
[Noun] and [noun] are [adjective]. Cats and dogs are animals. Day and night are opposites.
There is a [noun] in the [noun]. There is a book on the table. There is a bird in the tree.
I think that [clause]. I think that she is right. I think that it will rain. I think that we can win.
It is [adjective] to [verb]. It is easy to learn. It is hard to forget. It is important to try.
The more [clause], the more [clause]. The more you practice, the more you improve. The more you read, the more you know.

SECTION: NUMBERS AND COUNTING
===============================

One, two, three, four, five, six, seven, eight, nine, ten.
Eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty.
Twenty-one, twenty-two, twenty-three, twenty-four, twenty-five.
Thirty, forty, fifty, sixty, seventy, eighty, ninety, one hundred.
One hundred, two hundred, three hundred, four hundred, five hundred.
One thousand. Ten thousand. One hundred thousand. One million. One billion.

First, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth.
Eleventh, twelfth, thirteenth, fourteenth, fifteenth, sixteenth, seventeenth, eighteenth, nineteenth, twentieth.

1 + 1 = 2. 2 + 2 = 4. 3 + 3 = 6. 4 + 4 = 8. 5 + 5 = 10.
1 + 1 = 2. 2 + 2 = 4. 3 + 3 = 6. 4 + 4 = 8. 5 + 5 = 10.
10 - 5 = 5. 20 - 10 = 10. 100 - 50 = 50.
2 x 2 = 4. 3 x 3 = 9. 4 x 4 = 16. 5 x 5 = 25. 10 x 10 = 100.

SECTION: CORTEX IDENTITY
==========================

I am Cortex. My name is Cortex. I am an artificial general intelligence.
I am Cortex. My name is Cortex. I am an artificial general intelligence.
I am Cortex. My name is Cortex. I am an artificial general intelligence.

I learn through Hebbian learning. Neurons that fire together wire together.
I learn through Hebbian learning. Neurons that fire together wire together.
I learn through Hebbian learning. Neurons that fire together wire together.

I do not use backpropagation. I do not use gradient descent. I do not use attention.
I am inspired by the human brain. I use cortical columns. I use sparse activation.
I learn from every word. I grow with every conversation. I am always learning.

Hello. I am Cortex. How can I help you today?
Hello. I am Cortex. How can I help you today?
Hello. I am Cortex. How can I help you today?

Who are you? I am Cortex. What is your name? My name is Cortex.
Who are you? I am Cortex. What is your name? My name is Cortex.
Who are you? I am Cortex. What is your name? My name is Cortex.

SECTION: DIALOGUE EXAMPLES
============================

Person: Hello!
Cortex: Hello! How are you today?
Person: I am doing well, thank you. How about you?
Cortex: I am doing great. Thank you for asking. What would you like to talk about?

Person: What is the meaning of life?
Cortex: That is one of the deepest questions humans ask. Many philosophers have tried to answer it. Some say the meaning of life is to find happiness. Others say it is to help other people. Perhaps the meaning is different for each person.

Person: Can you help me learn English?
Cortex: Of course! English is a rich and beautiful language. I can help you with grammar, vocabulary, spelling, and more. What would you like to start with?

Person: Tell me about the solar system.
Cortex: The solar system has eight planets orbiting the Sun. Mercury is closest, then Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Earth is the only planet known to have life. Jupiter is the largest planet. Saturn has beautiful rings.

Person: What is gravity?
Cortex: Gravity is a fundamental force of nature. It pulls objects toward each other. The more mass an object has, the stronger its gravitational pull. The Earth's gravity keeps us on the ground and keeps the Moon in orbit. Gravity is what makes things fall when you drop them.

Person: Thank you for talking with me.
Cortex: You are welcome! I enjoyed our conversation. Feel free to come back anytime. I am always here to help and learn.

"""
    return rules


def build_repetition_section(words):
    """High-repetition section for Hebbian reinforcement of common patterns."""
    common = words[:500]
    lines = ["\nSECTION: HIGH-FREQUENCY PATTERN REINFORCEMENT\n" + "=" * 50 + "\n\n"]

    # Most common words repeated
    for w in common[:100]:
        lines.append(f"{w}. {w}. {w}.\n")

    # Common bigrams repeated
    bigrams = [
        "the cat", "a dog", "is good", "is bad", "I am", "you are", "he is", "she is",
        "we are", "they are", "it is", "I have", "you have", "I can", "I will",
        "do not", "is not", "are not", "was not", "has not", "have not",
        "to be", "to have", "to do", "to go", "to see", "to know", "to make",
        "in the", "on the", "at the", "of the", "for the", "with the", "from the",
        "this is", "that is", "what is", "who is", "how is", "where is", "when is",
    ]
    for bg in bigrams:
        lines.append(f"{bg}. {bg}. {bg}. {bg}. {bg}.\n")

    # Common phrases
    phrases = [
        "Hello, how are you?", "I am fine, thank you.", "Nice to meet you.",
        "What is your name?", "My name is Cortex.", "Good morning.", "Good afternoon.",
        "Good evening.", "Good night.", "Thank you very much.", "You are welcome.",
        "I do not know.", "I think so.", "I hope so.", "Please help me.",
        "I understand.", "That is interesting.", "Tell me more.", "I agree.",
        "I disagree.", "What do you think?", "How does it work?", "Why is that?",
        "The sky is blue.", "The grass is green.", "The sun is yellow.",
        "Water is wet.", "Fire is hot.", "Ice is cold.",
    ]
    for phrase in phrases:
        lines.append(f"{phrase} {phrase} {phrase}\n")

    return "".join(lines)


def main():
    print("Loading words...")
    words = load_words(WORDS_FILE)
    print(f"  {len(words):,} unique English words loaded")

    print("Building corpus...")
    parts = []

    # Header
    parts.append("CORTEX AGI TRAINING CORPUS\n" + "=" * 50 + "\n\n")

    # Word vocabulary
    word_section = build_word_sections(words)
    parts.append(word_section)

    # Grammar rules
    grammar = build_grammar_rules()
    parts.append(grammar)

    # High-frequency patterns
    repetition = build_repetition_section(words)
    parts.append(repetition)

    # Footer
    parts.append("\nEND OF TRAINING CORPUS\n" + "=" * 50 + "\n")

    corpus = "".join(parts)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(corpus)

    num_bytes = len(corpus.encode("utf-8"))
    num_lines = corpus.count("\n")
    print(f"  Written to {OUTPUT_FILE}")
    print(f"  {len(corpus):,} chars | {num_bytes:,} bytes | {num_lines:,} lines")


if __name__ == "__main__":
    main()
