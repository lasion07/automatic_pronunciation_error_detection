import eng_to_ipa as ei

def eng2ipa(text):
    ipa_text = ei.convert(text).replace('*','')
    return ipa_text