import os
import streamlit.web.bootstrap as bootstrap
import streamlit.runtime.scriptrunner.magic_funcs


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    flag_options = {
        "server.port": 8501,
        "global.developmentMode": False,
    }

    bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    bootstrap.run(
        "./home.py",
        "streamlit run",
        [],
        flag_options
        )