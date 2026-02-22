
class TACGlobalDecl:
    def __init__(self, name: str, init_vals: list[int], size: int, type: str) -> None:
        self.name = name
        self.init_vals = init_vals
        self.size = size
        self.type = type

    def printTo(self) -> None:
        if self.init_vals is not None:
            if self.type == "int":
                print("DECLARATION %s = %d" % (self.name, self.init_vals[0]))
            else:
                print("DECLARATION %s = {%s}" % (self.name, ', '.join(map(str, self.init_vals))))
        else:
            print("DECLARATION %s" % (self.name))
