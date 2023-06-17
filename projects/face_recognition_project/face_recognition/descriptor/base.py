class Descriptor():
    _name: str | None = None
    _default_params: dict | None = None

    def __init__(self, overlapParams: dict | None = None):
        self._params = overlapParams or self._default_params
        
        if self._params is None:
            print(f'INFO: {self.__class__.__name__} has no default parameters.')

    def __init_subclass__(cls) -> None:
        if cls._name is None:
            raise NotImplementedError(f'{cls.__name__} must implement the _name class attribute.')
        if cls._default_params is None:
            raise NotImplementedError(f'{cls.__name__} must implement the _default_params class attribute.')

    def describe(self, image):
        raise NotImplementedError()
    
    def __repr__(self):
        return f'{self.__class__.__name__}(params={self._params})'
    
    def __str__(self):
        return self.__repr__()