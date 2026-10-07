import dataclasses

import pytest
import yaml

from wildtrack import config

from .conftest import CONFIG


def raw():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_shipped_config_loads(cfg):
    assert cfg.controller.rate_hz == 10.0
    assert cfg.safety.min_battery_pct == 35.0
    assert cfg.link.ws_url == "ws://192.168.43.1:8765/link"
    assert not cfg.lock.auto_engage, "auto_engage must ship off"
    assert not cfg.classes.test_mode


def test_unknown_key_is_an_error():
    d = raw()
    d["lock"]["candidate_conff"] = 0.5
    with pytest.raises(config.ConfigError, match="unknown"):
        config.from_dict(d)


def test_missing_key_is_an_error():
    d = raw()
    del d["safety"]["min_gps_sats"]
    with pytest.raises(config.ConfigError, match="missing"):
        config.from_dict(d)


def test_wrong_type_is_an_error():
    d = raw()
    d["safety"]["min_gps_sats"] = "ten"
    with pytest.raises(config.ConfigError):
        config.from_dict(d)


def test_person_can_never_be_a_target():
    d = raw()
    d["classes"]["targets"].append("person")
    with pytest.raises(config.ConfigError, match="person"):
        config.from_dict(d)
    d = raw()
    d["classes"]["test_classes"] = ["person"]
    with pytest.raises(config.ConfigError, match="person"):
        config.from_dict(d)


def test_rate_outside_dji_range_is_an_error():
    d = raw()
    d["controller"]["rate_hz"] = 50.0
    with pytest.raises(config.ConfigError, match="5-25"):
        config.from_dict(d)


def test_kp_tau_warning():
    d = raw()
    d["controller"]["kp"] = 1.2  # 1.2 * 0.5 s = 0.6 rad > 0.4
    cfg = config.from_dict(d)
    assert any("kp*tau" in w for w in config.validate(cfg))


def test_override(cfg):
    c2 = config.override(cfg, "lock.auto_engage", True)
    assert c2.lock.auto_engage and not cfg.lock.auto_engage
    with pytest.raises(config.ConfigError):
        config.override(cfg, "lock.nope", 1)


def test_paths_resolve_relative_to_config(cfg):
    assert cfg.path(cfg.geo.dem_dir) == CONFIG.parent / "dem"
    assert dataclasses.is_dataclass(cfg)
