from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class FFT(Op):
    op = "FFT"
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(
            graph,
            {
                "type": __class__.op,
                "op": __class__.op,
                "in_ports_count": 2,
                "out_ports_count": 1,
                "infer": copy_shape_infer,
            },
            attrs,
        )

    def supported_attrs(self):
        return ["inverse", "centered"]


class IFFT(Op):
    op = "IFFT"
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(
            graph,
            {
                "type": __class__.op,
                "op": __class__.op,
                "in_ports_count": 2,
                "out_ports_count": 1,
                "infer": copy_shape_infer,
            },
            attrs,
        )

    def supported_attrs(self):
        return ["inverse", "centered"]
