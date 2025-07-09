import logging
from typing import Optional, Dict, Any


class Base:
    def __init__(
        self, params: Optional[Dict[str, Any]] = None, log_level: int = logging.DEBUG
    ):
        self.params = params
        self.log_level = log_level
        self._set_logger(self.log_level)

    def _set_logger(self, level: int):
        # re-configure root handlers
        logging.basicConfig(
            level=level,
            format=" %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("/tmp/debug.log"), logging.StreamHandler()],
            force=True,
        )
        # get a module-level logger, then set its level too
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        print(f"Logger initialized with level: {logging.getLevelName(level)}")
