import logging
from hf_model_serving.utils import load_config
from hf_model_serving.model_registration import ModelRegistrar
from hf_model_serving.model_serving import ModelServingDeployer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

def deploy_main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    model_cfg = cfg["model"]
    serving_cfg = cfg["serving"]

    registrar = ModelRegistrar(model_cfg)
    model_version = registrar.register_model()

    deployer = ModelServingDeployer(model_cfg, serving_cfg, model_version)
    deployer.deploy()


if __name__ == "__main__":
    deploy_main()
