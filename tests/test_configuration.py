from swtaudiofakedetect.configuration import Config


class TestConfiguration:
    def test_load_config_from_file(self) -> None:
        # TODO
        pass

    def test_load_config_from_string(self) -> None:
        yaml = """dataset_dir: ./datasets/debug/
batch_size: 256
num_workers: 6
tasks:
  -
    main_module: src.models.test1
    output_dir: ./out/test1/
    stop_epoch: 10
    num_checkpoints: 0
    num_validations: 4
    batch_size: 128
  -
    main_module: src.models.test2
    output_dir: ./out/test2/
    stop_epoch: 20
    num_checkpoints: 5
    num_validations: 10
    weighted_loss: true"""
        cfg = Config(yaml)

        assert cfg.num_tasks() == 2

        module1, kwargs1 = cfg.get_task(0)

        assert module1 == "src.models.test1"
        assert kwargs1["output_dir"] == "./out/test1/"
        assert kwargs1["stop_epoch"] == 10
        assert kwargs1["batch_size"] == 128
        assert kwargs1["num_workers"] == 6

        module2, kwargs2 = cfg.get_task(1)

        assert module2 == "src.models.test2"
        assert kwargs2["output_dir"] == "./out/test2/"
        assert kwargs2["stop_epoch"] == 20
        assert kwargs2["batch_size"] == 256
        assert kwargs2["num_workers"] == 6
        assert kwargs2["weighted_loss"] == True
