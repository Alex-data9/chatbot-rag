"""
Detecção automática do perfil do lead com base nas mensagens trocadas.
O perfil é usado para personalizar a abordagem do agente.
"""
from enum import Enum


class LeadProfile(str, Enum):
    VESTIBULANDO = "vestibulando"
    PROFISSIONAL = "profissional"
    TRANSFERIDO = "transferido"
    POS_GRADUANDO = "pos_graduando"
    PAI_MAE = "pai_mae"
    OUTRO = "outro"


# Palavras-chave por perfil (em minúsculas, sem acento para maior cobertura)
_KEYWORDS: dict[LeadProfile, list[str]] = {
    LeadProfile.VESTIBULANDO: [
        "vestibular", "ensino medio", "primeiro curso", "recém formado",
        "recem formado", "acabei de sair", "tenho 17", "tenho 18", "tenho 19",
        "nunca fiz faculdade", "primeiro ingresso",
    ],
    LeadProfile.PROFISSIONAL: [
        "ja trabalho", "já trabalho", "emprego", "mudança de carreira",
        "mudanca de carreira", "requalificacao", "requalificação",
        "crescer profissionalmente", "promoção", "promocao", "salario",
        "salário", "noturno", "ead", "trabalho de dia",
    ],
    LeadProfile.TRANSFERIDO: [
        "transferencia", "transferência", "outra faculdade", "outra universidade",
        "aproveitamento de disciplinas", "trancado", "trancada",
        "ja cursei", "já cursei", "quero mudar de faculdade",
    ],
    LeadProfile.POS_GRADUANDO: [
        "pos-graduacao", "pós-graduação", "especializacao", "especialização",
        "mestrado", "mba", "ja me formei", "já me formei", "sou graduado",
        "sou formado", "quero me especializar",
    ],
    LeadProfile.PAI_MAE: [
        "meu filho", "minha filha", "para meu filho", "para minha filha",
        "sou pai", "sou mae", "sou mãe", "estou buscando para",
        "quero matricular meu", "quero matricular minha",
    ],
}


def _normalize(text: str) -> str:
    """Remove acentos e converte para minúsculas para comparação robusta."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def detect_profile(message: str) -> LeadProfile:
    """
    Infere o perfil do lead a partir de uma mensagem.
    Retorna LeadProfile.OUTRO se nenhum padrão for reconhecido.
    """
    normalized = _normalize(message)
    for profile, keywords in _KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            return profile
    return LeadProfile.OUTRO


def detect_profile_from_history(messages: list[str]) -> LeadProfile:
    """
    Analisa todo o histórico de mensagens do usuário para detectar o perfil.
    Útil quando a primeira mensagem não é suficientemente informativa.
    """
    combined = " ".join(messages)
    return detect_profile(combined)


PROFILE_LABELS: dict[LeadProfile, str] = {
    LeadProfile.VESTIBULANDO: "Vestibulando (primeiro curso)",
    LeadProfile.PROFISSIONAL: "Profissional em transição/requalificação",
    LeadProfile.TRANSFERIDO: "Transferência de outra instituição",
    LeadProfile.POS_GRADUANDO: "Graduado buscando pós-graduação",
    LeadProfile.PAI_MAE: "Pai/Mãe buscando para filho(a)",
    LeadProfile.OUTRO: "Perfil não identificado",
}
