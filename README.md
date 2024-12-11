# cs-7643-project

**HOW TO RUN:**
- Create conda environment:
  `conda env create -f environment.yml`
- Activate conda environment:
  `conda activate cs7643_project`
- Download Spacy model 
 `python -m spacy download en_core_web_sm`
- Change the configuration file:
  `config/config_*.json`
- Update the path to the config file in the setting.py:
  `app/settings.py:119`
- Run the project
  `python main.py`
