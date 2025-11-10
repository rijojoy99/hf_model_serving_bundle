hf_model_serving_bundle/
├── bundle.yml
├── config.yaml
├── init.sh
├── setup.py
└── hf_model_serving/
    ├── __init__.py
    ├── bootstrap_unity_catalog.py
    ├── deploy_main.py
    ├── model_registration.py
    ├── model_serving.py
    └── utils.py





Local run - 

python hf_model_serving/deploy_main.py --config_path config.yaml

Deploy via DAB job

databricks bundle deploy
databricks bundle run hf_model_serving_job
