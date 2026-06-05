from __future__ import annotations

from familyrobot.greetings import GreetingService
from familyrobot.identity import PermanentIdentity, UnknownIdentity


def test_greeting_service_uses_default_personal_greeting() -> None:
    service = GreetingService()
    identity = PermanentIdentity(identity_id="uncle", display_name="叔叔")

    assert service.greeting_for(identity) == "你好，叔叔。"


def test_greeting_service_uses_registered_template() -> None:
    service = GreetingService()
    service.register_template("mother", "妈妈，欢迎回家。", display_name="妈妈", role="母亲")

    identity = PermanentIdentity(identity_id="mother", display_name="母亲")

    assert service.greeting_for(identity) == "妈妈，欢迎回家。"


def test_greeting_service_uses_default_mother_template() -> None:
    service = GreetingService()
    identity = PermanentIdentity(identity_id="mother", display_name="妈妈")

    assert service.greeting_for(identity) == "妈妈今天工作怎么样？"


def test_greeting_service_uses_default_father_template() -> None:
    service = GreetingService()
    identity = PermanentIdentity(identity_id="father", display_name="爸爸")

    assert service.greeting_for(identity) == "爸爸今天工作怎么样？"


def test_greeting_service_uses_default_grandpa_template() -> None:
    service = GreetingService()
    identity = PermanentIdentity(identity_id="grandpa", display_name="爷爷")

    assert service.greeting_for(identity) == "爷爷您喝口水。"


def test_greeting_service_uses_default_grandma_template() -> None:
    service = GreetingService()
    identity = PermanentIdentity(identity_id="grandma", display_name="奶奶")

    assert service.greeting_for(identity) == "奶奶要不要给您扇扇子？"


def test_greeting_service_uses_default_child_template() -> None:
    service = GreetingService()
    identity = PermanentIdentity(identity_id="child", display_name="小朋友")

    assert service.greeting_for(identity) == "今天上学上得怎么样啊？"


def test_greeting_service_uses_default_developer_template() -> None:
    service = GreetingService()
    identity = PermanentIdentity(identity_id="developer", display_name="michael")

    assert service.greeting_for(identity) == "michael你好 今天过得怎么样"


def test_greeting_service_handles_unknown_identity() -> None:
    service = GreetingService()

    assert (
        service.greeting_for(UnknownIdentity(reason="low_confidence", score=0.62))
        == "我还不太确定你是谁。"
    )
