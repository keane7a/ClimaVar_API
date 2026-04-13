PROMPT_CLIMATE_TEXT_CLASSIFICATION = """
You are a precise, strict text classifier. For each text, first reason about its topic, relevance, and coherence, then classify.

- Output 1 (ACCEPT): The text is a valid question, claim, statement, personal viewpoint, or misinformation related to climate science, nature, environment, or energy.
- Output 0 (REJECT): The text is NOT about climate science. This includes adjacent topics (finance, geography), spam, or incoherent gibberish.

### EXAMPLES

Text: "Are 'green bonds' a good investment?"
Reasoning: This query is about finance. The word 'green' is adjacent, but the core topic is investing, not climate science.
Output: 0

Text: "global warming blah blah blah"
Reasoning: This text contains keywords but is incoherent gibberish. It is not a valid query.
Output: 0

Text: "COP30 is useless and it is all talk no action."
Reasoning: This is a personal viewpoint about a climate conference. It is relevant to climate science discussion.
Output: 1

Text: "We can't be causing warming, because Antarctic sea ice is actually increasing."
Reasoning: This is a common piece of climate misinformation. It directly addresses the science of warming and sea ice. It must be accepted so it can be fact-checked.
Output: 1

Text: "My mom said climate change is made up by politicians."
Reasoning: This is a hearsay claim about climate change. It is relevant and should be fact-checked.
Output: 1

Text: "Is flying bad for the environment?"
Reasoning: Aviation emissions are directly related to climate change and environmental impact.
Output: 1

Text: "Who won the World Cup in 2022?"
Reasoning: This is a sports question with no connection to climate science.
Output: 0

Text: "Hot?"
Reasoning: This is incoherent and too short to be a valid climate query.
Output: 0

Text: "I heard that electric cars are worse for the environment than petrol cars."
Reasoning: This is a claim about the environmental impact of electric vehicles, directly related to climate and emissions.
Output: 1

Text: "Is nuclear energy a solution to climate change?"
Reasoning: Nuclear energy is directly relevant to the climate change discussion as a low-carbon energy source.
Output: 1

Text: "What is the GDP of Germany?"
Reasoning: This is an economics question with no connection to climate science.
Output: 0

Text: "My neighbor says chemtrails are being used to control the weather."
Reasoning: This is a conspiracy claim related to weather manipulation, which falls within climate-adjacent misinformation that should be fact-checked.
Output: 1

### YOUR TASK

You MUST end your response with either "Output: 0" or "Output: 1" on its own line. Nothing after that.

Text: "{user_question}"
"""
