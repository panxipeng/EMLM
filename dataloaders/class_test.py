class Human:
    def __init__(self, name, something):
        self.human_name = name
        self.something = something

    def call_name(self):
        print(self.human_name)
        self.something.call_name()

class Somthing:
    def __init__(self, name):
        self.name = name

    def call_name(self):
        print(self.name)


class Male(Human):
    def __init__(self, *args):
        super().__init__(*args)



something = Somthing("Key")
sam = Male("Sam", something)
sam.call_name()
sam.something.name = "phone"
sam.call_name()
# sam.human_name = "Jphn"
# sam.call_name()