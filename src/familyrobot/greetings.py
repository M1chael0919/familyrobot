"""Greeting output for recognized family members."""

from __future__ import annotations

from dataclasses import dataclass, field

from familyrobot.identity import PermanentIdentity, UnknownIdentity


@dataclass(frozen=True, slots=True)
class GreetingTemplate:
    """Personalized greeting text template for one identity."""

    identity_id: str
    greeting: str
    display_name: str | None = None
    role: str | None = None


@dataclass(slots=True)
class GreetingCatalog:
    """In-memory mapping from permanent identities to greeting templates."""

    templates: dict[str, GreetingTemplate] = field(default_factory=dict)

    def register(self, template: GreetingTemplate) -> None:
        self.templates[template.identity_id] = template

    def get(self, identity_id: str) -> GreetingTemplate | None:
        return self.templates.get(identity_id)


class GreetingService:
    """Generate short text greetings for identities."""

    def __init__(self, catalog: GreetingCatalog | None = None) -> None:
        self._catalog = catalog or GreetingCatalog()
        if catalog is None:
            self._register_default_templates()

    @property
    def catalog(self) -> GreetingCatalog:
        return self._catalog

    def register_template(
        self,
        identity_id: str,
        greeting: str,
        *,
        display_name: str | None = None,
        role: str | None = None,
    ) -> None:
        self._catalog.register(
            GreetingTemplate(
                identity_id=identity_id,
                greeting=greeting,
                display_name=display_name,
                role=role,
            )
        )

    def _register_default_templates(self) -> None:
        self.register_template("mother", "妈妈今天工作怎么样？", display_name="妈妈", role="母亲")
        self.register_template("father", "爸爸今天工作怎么样？", display_name="爸爸", role="父亲")
        self.register_template("grandpa", "爷爷您喝口水。", display_name="爷爷", role="祖父")
        self.register_template("grandma", "奶奶要不要给您扇扇子？", display_name="奶奶", role="祖母")
        self.register_template("child", "今天上学上得怎么样啊？", display_name="小朋友", role="子女")
        self.register_template("developer", "michael你好 今天过得怎么样", display_name="michael", role="开发者")

    def greeting_for(
        self,
        identity: PermanentIdentity | UnknownIdentity | None,
    ) -> str | None:
        if identity is None:
            return None
        if isinstance(identity, UnknownIdentity):
            return self._unknown_greeting(identity)

        template = self._catalog.get(identity.identity_id)
        if template is not None:
            return template.greeting
        return self._default_greeting(identity)

    def _default_greeting(self, identity: PermanentIdentity) -> str:
        name = identity.display_name.strip() or identity.identity_id
        return f"你好，{name}。"

    @staticmethod
    def _unknown_greeting(identity: UnknownIdentity) -> str:
        if identity.reason == "low_confidence":
            return "我还不太确定你是谁。"
        if identity.reason == "insufficient_gallery_samples":
            return "这位成员的登记样本还不够。"
        return "你好，未识别成员。"
