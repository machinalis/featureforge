import schema

class Feature(object):

    input_schema = schema.Schema(object)
    output_schema = schema.Schema(object)

    def __call__(self, data_point):
        try:
            data_point = self.input_schema.validate(data_point)
        except schema.SchemaError as e:
            raise ValueError(e)
        result = self._evaluate(data_point)
        try:
            return self.output_schema.validate(result)
        except schema.SchemaError as e:
            raise ValueError(e)

    def _evaluate(self, data_point):
        return None


