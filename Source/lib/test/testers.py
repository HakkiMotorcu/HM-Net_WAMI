from .base_tester import BaseTester

testers={
            "base":BaseTester,
}

def get_tester(tag,default="base"):
    tag=tag.lower().replace("_","")
    if isinstance(tag,str) and tag in testers.keys():
        return testers[tag]
    else:
        return testers[tag.lower().replace("_","")]
