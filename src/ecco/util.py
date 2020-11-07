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