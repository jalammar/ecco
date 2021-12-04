import yaml
import os

# CHeck if running from inside jupyter
# From https://stackoverflow.com/questions/47211324/check-if-module-is-running-in-jupyter-or-not
def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'

def load_config(model_name):
    path = os.path.dirname(__file__) 
    configs = yaml.safe_load(open(os.path.join(path, "model-config.yaml"),
                                  encoding="utf8"))
    try:
        model_config = configs[model_name]
        tokenizer_config = {'token_prefix': model_config['token_prefix'],
                           'partial_token_prefix': model_config['partial_token_prefix']}
        model_config['tokenizer_config'] = tokenizer_config
    except KeyError:
        raise ValueError(
                f"The model '{model_name}' is not defined in Ecco's 'model-config.yaml' file and"
                f" so is not explicitly supported yet. Supported models are:",
                list(configs.keys())) from KeyError()
    return model_config