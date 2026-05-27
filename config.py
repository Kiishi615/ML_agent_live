from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


@dataclass(frozen=True)
class LoggingConfig:
    level:int
    log_dir :str 

    def __post_init__(self) -> None:
        """
        
        """
        if not self.log_dir.strip():
            raise ConfigError("Specify Logging directory please")

        if self.level not in [10, 20, 30, 40, 50]:
            raise ConfigError(
                f"logging level must be in [10, 20, 30, 40, 50] got: {self.level}"
            )


@dataclass(frozen=True)
class DatabaseConfig:

    name: str
    directory: str
    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("database.host cannot be empty")
        if not self.directory.strip():
            raise ConfigError("database.name cannot be empty")

    @property
    def connection_string(self) -> str:
        return (
            f"sqlite:///{self.directory}/{self.name}"
            
        )


@dataclass(frozen=True)
class AppConfig:
    
    logging: LoggingConfig
    database: DatabaseConfig
    environment: str

    def __post_init__(self) -> None:
        valid_envs = {"dev", "test", "staging", "prod"}
        if self.environment not in valid_envs:
            raise ConfigError(
                f"environment must be one of {valid_envs}, "
                f"got '{self.environment}'"
            )


def _get_value(
    parser: configparser.ConfigParser,
    section: str,
    key: str,
    env_prefix: str = "APP",
    ) -> str:
    
    env_var = f"{env_prefix}_{section}_{key}".upper()

    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value

    try:
        return parser.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ConfigError(
            f"Missing config: [{section}] {key} "
            f"(also checked env var {env_var})"
        ) from e


def load_config(
    env: str | None = None,
    config_dir: Path = Path("configs"),
    ) -> AppConfig:

    if env is None:
        env = os.environ.get("APP_ENV", "dev")

    config_path = config_dir / f"{env}.cfg"

    if not config_path.exists():
        available = [f.name for f in config_dir.glob("*.cfg")]
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            f"Available: {available}"
        )

    parser = configparser.ConfigParser()
    read_ok = parser.read(config_path)
    if not read_ok:
        raise ConfigError(f"Failed to parse: {config_path}")

    def get(section: str, key: str) -> str:
        return _get_value(parser, section, key)

    def get_int(section: str, key: str) -> int:
        raw = get(section, key).strip()
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(
                f"[{section}] {key} = '{raw}' is not a valid integer"
            )

    def get_bool(section: str, key: str) -> bool:
        raw = get(section, key).lower()
        if raw in ("true", "yes", "on", "1"):
            return True
        if raw in ("false", "no", "off", "0"):
            return False
        raise ConfigError(
            f"[{section}] {key} = '{raw}' is not a valid boolean"
        )

    logging = LoggingConfig(
        level=get_int("logging", "level"),
        log_dir=get("logging", "log_dir"),
    )

    database = DatabaseConfig(
        name=get("database", "name"),
        directory=get("database", "directory"),

    )
    return AppConfig(
        logging=logging,
        database=database,
        environment=env,
        
    )        
    