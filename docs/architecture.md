Ecco is made up of two components:

- [Ecco](https://github.com/jalammar/ecco), a python component. Wraps around language models and collects relevant data. 
- [EccoJS](https://github.com/jalammar/eccojs), a Javascript component used to create interactive explorables from the outputs of Ecco.

All the machine learning happens in the Ecco. The results can be plotted by python, or interactive explorables are created using eccoJS.

Ecco's major components are:

- LM -- a class that wraps around a language model. It has hooks into HuggingFace Transformers language models which allow collecting data (like neuron activations) and making useful calculations (like input saliency). LM is the main component users interact with in Ecco. This is usually by loading a model (i.e. `lm = ecco.from_pretrained('gpt2)`), then processing some input text using the model:

    input_text = "Hello, machine"
    output = lm.generate(input_text)
      
    # or
    inputs = lm.tokenizer(input_text)
    output = lm(inputs)
- OutputSeq -- 

