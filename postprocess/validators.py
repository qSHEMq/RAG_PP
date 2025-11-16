import re

def valid_date(s: str) -> bool:
    return bool(re.match(r"^(0?[1-9]|[12]\d|3[01])([.\s])?(0?[1-9]|1[0-2])\2(20\d{2})$", s))

def norm_date(s: str) -> str:
    m = re.match(r"^(0?[1-9]|[12]\d|3[01])([.\s])?(0?[1-9]|1[0-2])\2(20\d{2})$", s)
    if not m: return s
    d, _, mth, yr = m.group(1), m.group(2), m.group(3), m.group(4)
    return f"{int(d):02d}.{int(mth):02d}.{yr}"

def valid_inn10(s: str) -> bool:
    if not re.fullmatch(r"\d{10}", s): return False
    d = list(map(int, s))
    k = (2*d[0]+4*d[1]+10*d[2]+3*d[3]+5*d[4]+9*d[5]+4*d[6]+6*d[7]+8*d[8])%11%10
    return k == d[9]

def valid_inn12(s: str) -> bool:
    if not re.fullmatch(r"\d{12}", s): return False
    d = list(map(int, s))
    k1 = (7*d[0]+2*d[1]+4*d[2]+10*d[3]+3*d[4]+5*d[5]+9*d[6]+4*d[7]+6*d[8]+8*d[9])%11%10
    k2 = (3*d[0]+7*d[1]+2*d[2]+4*d[3]+10*d[4]+3*d[5]+5*d[6]+9*d[7]+4*d[8]+6*d[9]+8*d[10])%11%10
    return (k1==d[10]) and (k2==d[11])

def valid_kpp(s: str) -> bool:
    return bool(re.fullmatch(r"\d{9}", s))

def valid_ogrn(s: str) -> bool:
    if not re.fullmatch(r"\d{13}", s): return False
    return int(s[-1]) == (int(s[:-1]) % 11) % 10

def valid_ogrnip(s: str) -> bool:
    if not re.fullmatch(r"\d{15}", s): return False
    return int(s[-1]) == (int(s[:-1]) % 13) % 10

def norm_amount(s: str) -> str:
    t = s.replace(" ", "").replace("\u2009", "").replace("\u00A0", "")
    t = t.replace(",", ".")
    m = re.match(r"^\d+(\.\d+)?$", t)
    return t if m else s
