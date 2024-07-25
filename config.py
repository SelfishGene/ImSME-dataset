# Configuration parameters and fixed data dictionaries/lists for ImSME dataset generation

# Character replacement dictionary
CHAR_REPLACEMENT_DICT = {
    '"': 'doublequote',
    "'": 'singlequote',
    '\\': 'backslash',
    '/': 'forwardslash',
    ':': 'colon',
    '*': 'asterisk',
    '?': 'questionmark',
    '<': 'lessthan',
    '>': 'greaterthan',
    '|': 'pipe',
    ' ': 'space',
    '(': 'leftparenthesis',
    ')': 'rightparenthesis',
    '[': 'leftbracket',
    ']': 'rightbracket',
    '{': 'leftbrace',
    '}': 'rightbrace',
    '#': 'hash',
    '%': 'percent',
    '&': 'ampersand',
    '^': 'caret',
    '@': 'at',
    ';': 'semicolon',
    ',': 'comma',
    '.': 'dot',
    '`': 'backtick',
    '=': 'equal',
    '!': 'exclamation',
    '+': 'plus',
    '-': 'minus',
    '_': 'underscore',
    '$': 'dollar',
    '~': 'tilde',
}

# Reverse character replacement dictionary
REVERSE_CHAR_REPLACEMENT_DICT = {v: k for k, v in CHAR_REPLACEMENT_DICT.items()}

# Character lists
BASE_ENGLISH_CHARACTERS_LIST = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&()*+-/:;<=>?@[\\]^{|}')
CHARACTERS_LIST = BASE_ENGLISH_CHARACTERS_LIST

# Char order for label generation
CHAR_ORDER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=', ' ']

# Section order for label generation
SECTION_ORDER = ["argument 1", "operation", "argument 2", "equal sign", "result", "digit", "symbol", "space"]

# Font order for label generation
FONT_ORDER = [
    'ARIALN', 'times', 'verdanaz', 'georgiai', 'couri', 'tahomabd', 'impact', 'comicz',
    'calibriz', 'GARA', 'pala', 'BOOKOSBI', 'CENTURY', 'trebucbi', 'GILSANUB', 'BASKVILL',
    'ROCKEB', 'FRADMCN', 'cambriaz', 'consola', 'COPRGTB', 'PAPYRUS', 'BRUSHSCI', 'lucon',
    'BAUHS93', 'BOD_BLAI', 'AGENCYR', 'ALGER', 'BELLI', 'BRLNSB', 'BRITANIC', 'BROADW',
    'CALIFR', 'CALISTB', 'CASTELAR', 'CENTAUR', 'CHILLER', 'COLONNA', 'COOPBL', 'ELEPHNTI',
    'ENGR', 'ERASBD', 'FELIXTI', 'FORTE', 'FREESCPT', 'GIGI', 'GOUDOSI', 'HATTEN',
    'HARLOWSI', 'HTOWERT', 'JOKERMAN', 'KUNSTLER', 'LCALLIG', 'MAGNETOB', 'MAIAN', 'MATURASC',
    'MISTRAL', 'MOD20', 'MTCORSVA', 'NIAGENG', 'ONYX', 'PLAYBILL', 'POORICH', 'PRISTINA',
    'RAGE', 'RAVIE', 'SCRIPTBL', 'SHOWG', 'STENCIL', 'TEMPSITC', 'VINERITC', 'VIVALDII',
    'VLADIMIR', 'seguiemj', 'Candara', 'corbelb', 'Gabriola', 'Inkfree', 'javatext', 'LEELAWDB',
    'micross', 'mvboli', 'sylfaen', 'BERNHC', 'BKANT', 'BRADHITC', 'BRLNSDB', 'COPRGTL',
    'DUBAI-LIGHT', 'ERASDEMI', 'ERASLGHT', 'FRABKIT', 'FRAHV', 'FRAMDCN', 'FTLTLT', 'GLECB',
    'GOTHICI', 'HARNGTON', 'IMPRISHA', 'ITCBLKAD', 'ITCKRIST', 'LATINWD', 'LBRITEDI', 'LHANDW'
]


# Section name mapping
SECTION_NAME_MAPPING = {
    'argument 1': 'argument1',
    'operation': 'operation',
    'argument 2': 'argument2',
    'equal sign': 'equal_sign',
    'result': 'result'
}

# Operations
OPERATIONS = ['+', '-', '*', '/']

# Symbol characters
SYMBOL_CHARS = set(['+', '-', '*', '/', '='])

# Digit characters
DIGIT_CHARS = set('0123456789')

# Font name to short path dictionary
NAME_SHORT_PATH_DICT = {
    'Franklin Gothic Demi': 'FRADM.TTF',
    'Goudy Old Style': 'GOUDOSI.TTF',
    'Cambria': 'cambriaz.ttf',
    'David': 'davidbd.ttf',
    'Informal Roman': 'INFROMAN.TTF',
    'Dubai': 'DUBAI-LIGHT.TTF',
    'Jokerman': 'JOKERMAN.TTF',
    'Broadway': 'BROADW.TTF',
    'Agency FB': 'AGENCYR.TTF',
    'Eras Bold ITC': 'ERASBD.TTF',
    'Franklin Gothic Heavy': 'FRAHV.TTF',
    'Times New Roman': 'times.ttf',
    'Viner Hand ITC': 'VINERITC.TTF',
    'Tahoma': 'tahomabd.ttf',
    'Stencil': 'STENCIL.TTF',
    'Chiller': 'CHILLER.TTF',
    'Segoe UI': 'segoeuil.ttf',
    'Rockwell': 'ROCK.TTF',
    'Trebuchet MS': 'trebucbi.ttf',
    'Lucida Fax': 'LFAXD.TTF',
    'Calisto MT': 'CALISTB.TTF',
    'Footlight MT Light': 'FTLTLT.TTF',
    'Bodoni MT': 'BOD_BLAI.TTF',
    'Microsoft PhagsPa': 'phagspa.ttf',
    'Franklin Gothic Book': 'FRABKIT.TTF',
    'Book Antiqua': 'BKANT.TTF',
    'Segoe Print': 'segoepr.ttf',
    'Century Gothic': 'GOTHICI.TTF',
    'Constantia': 'constani.ttf',
    'Bradley Hand ITC': 'BRADHITC.TTF',
    'Britannic Bold': 'BRITANIC.TTF',
    'Gill Sans Ultra Bold Condensed': 'GILLUBCD.TTF',
    'Lucida Bright': 'LBRITEDI.TTF',
    'Segoe Script': 'segoescb.ttf',
    'Ebrima': 'ebrimabd.ttf',
    'Century': 'CENTURY.TTF',
    'Berlin Sans FB': 'BRLNSB.TTF',
    'Calibri': 'calibriz.ttf',
    'Hadassah Friedlaender': 'HADASAH.TTF',
    'Palatino Linotype': 'pala.ttf',
    'Rockwell Condensed': 'ROCC____.TTF',
    'Lucida Sans': 'LSANS.TTF',
    'Candara': 'Candara.ttf',
    'Yu Gothic': 'YuGothM.ttc',
    'Lucida Sans Typewriter': 'LTYPEO.TTF',
    'Nirmala UI': 'NirmalaB.ttf',
    'Garamond': 'GARA.TTF',
    'Microsoft Tai Le': 'taile.ttf',
    'Perpetua': 'PERB____.TTF',
    'Californian FB': 'CALIFR.TTF',
    'Bookman Old Style': 'BOOKOSBI.TTF',
    'Microsoft JhengHei': 'msjh.ttc',
    'French Script MT': 'FRSCRIPT.TTF',
    'Palace Script MT': 'PALSCRI.TTF',
    'Aharoni': 'ahronbd.ttf',
    'Gigi': 'GIGI.TTF',
    'Gill Sans Ultra Bold': 'GILSANUB.TTF',
    'Comic Sans MS': 'comicz.ttf',
    'Courier New': 'couri.ttf',
    'Leelawadee': 'LEELAWDB.TTF',
    'Corbel': 'corbelb.ttf',
    'Elephant': 'ELEPHNTI.TTF',
    'Verdana': 'verdanaz.ttf',
    'Sitka': 'SitkaVF.ttf',
    'Segoe UI Emoji': 'seguiemj.ttf',
    'Leelawadee UI': 'LeelaUIb.ttf',
    'Bell MT': 'BELLI.TTF',
    'Georgia': 'georgiai.ttf',
    'Monotype Corsiva': 'MTCORSVA.TTF',
    'SimSun-ExtB': 'simsunb.ttf',
    'Consolas': 'consola.ttf',
    'Lucida Calligraphy': 'LCALLIG.TTF',
    'Lucida Handwriting': 'LHANDW.TTF',
    'Arial': 'ARIALN.TTF',
    'Levenim MT': 'lvnmbd.ttf',
    'Pristina': 'PRISTINA.TTF',
    'Gill Sans MT Condensed': 'GILC____.TTF',
    'Copperplate Gothic Light': 'COPRGTL.TTF',
    'Gill Sans MT': 'GIL_____.TTF',
    'High Tower Text': 'HTOWERT.TTF',
    'Segoe UI Variable': 'SegUIVar.ttf',
    'Bahnschrift': 'bahnschrift.ttf',
    'Vladimir Script': 'VLADIMIR.TTF',
    'SimSun': 'simsun.ttc',
    'Arial Rounded MT Bold': 'ARLRDBD.TTF',
    'MS Reference Sans Serif': 'REFSAN.TTF',
    'Microsoft YaHei': 'msyh.ttc',
    'Segoe UI Historic': 'seguihis.ttf',
    'Microsoft Himalaya': 'himalaya.ttf',
    'Algerian': 'ALGER.TTF',
    'Perpetua Titling MT': 'PERTIBD.TTF',
    'Harlow Solid Italic': 'HARLOWSI.TTF',
    'Niagara Solid': 'NIAGSOL.TTF',
    'Gisha': 'gishabd.ttf',
    'Centaur': 'CENTAUR.TTF',
    'Rockwell Extra Bold': 'ROCKEB.TTF',
    'Eras Medium ITC': 'ERASMD.TTF',
    'Rod': 'rod.ttf',
    'Kristen ITC': 'ITCKRIST.TTF',
    'Tw Cen MT': 'TCM_____.TTF',
    'Microsoft Uighur': 'MSUIGHUR.TTF',
    'Century Schoolbook': 'SCHLBKI.TTF',
    'Niagara Engraved': 'NIAGENG.TTF',
    'Playbill': 'PLAYBILL.TTF',
    'Juice ITC': 'JUICE___.TTF',
    'Forte': 'FORTE.TTF',
    'Segoe UI Symbol': 'seguisym.ttf',
    'Ink Free': 'Inkfree.ttf',
    'Blackadder ITC': 'ITCBLKAD.TTF',
    'Edwardian Script ITC': 'ITCEDSCR.TTF',
    'Sylfaen': 'sylfaen.ttf',
    'Kunstler Script': 'KUNSTLER.TTF',
    'Tw Cen MT Condensed': 'TCCB____.TTF',
    'Gabriola': 'Gabriola.ttf',
    'Eras Light ITC': 'ERASLGHT.TTF',
    'Tw Cen MT Condensed Extra Bold': 'TCCEB.TTF',
    'Wide Latin': 'LATINWD.TTF',
    'Mistral': 'MISTRAL.TTF',
    'Vivaldi': 'VIVALDII.TTF',
    'Engravers MT': 'ENGR.TTF',
    'Berlin Sans FB Demi': 'BRLNSDB.TTF',
    'Impact': 'impact.ttf',
    'Castellar': 'CASTELAR.TTF',
    'Microsoft Yi Baiti': 'msyi.ttf',
    'Snap ITC': 'SNAP____.TTF',
    'Lucida Sans Unicode': 'l_10646.ttf',
    'Eras Demi ITC': 'ERASDEMI.TTF',
    'Brush Script MT': 'BRUSHSCI.TTF',
    'Gill Sans MT Ext Condensed Bold': 'GLSNECB.TTF',
    'Rage Italic': 'RAGE.TTF',
    'Bauhaus 93': 'BAUHS93.TTF',
    'Script MT Bold': 'SCRIPTBL.TTF',
    'Myanmar Text': 'mmrtextb.ttf',
    'Mongolian Baiti': 'monbaiti.ttf',
    'Franklin Gothic Demi Cond': 'FRADMCN.TTF',
    'Microsoft New Tai Lue': 'ntailub.ttf',
    'Franklin Gothic Medium': 'framd.ttf',
    'Goudy Stout': 'GOUDYSTO.TTF',
    'Tempus Sans ITC': 'TEMPSITC.TTF',
    'Matura MT Script Capitals': 'MATURASC.TTF',
    'Cooper Black': 'COOPBL.TTF',
    'Baskerville Old Face': 'BASKVILL.TTF',
    'Ravie': 'RAVIE.TTF',
    'MingLiU-ExtB': 'mingliub.ttc',
    'Imprint MT Shadow': 'IMPRISHA.TTF',
    'Microsoft Sans Serif': 'micross.ttf',
    'Franklin Gothic Medium Cond': 'FRAMDCN.TTF',
    'Showcard Gothic': 'SHOWG.TTF',
    'Narkisim': 'nrkis.ttf',
    'Colonna MT': 'COLONNA.TTF',
    'Bernard MT Condensed': 'BERNHC.TTF',
    'Modern No. 20': 'MOD20.TTF',
    'Felix Titling': 'FELIXTI.TTF',
    'Freestyle Script': 'FREESCPT.TTF',
    'Copperplate Gothic Bold': 'COPRGTB.TTF',
    'OCR A Extended': 'OCRAEXT.TTF',
    'Miriam': 'mriam.ttf',
    'Gloucester MT Extra Condensed': 'GLECB.TTF',
    'Harrington': 'HARNGTON.TTF',
    'Maiandra GD': 'MAIAN.TTF',
    'Lucida Console': 'lucon.ttf',
    'Haettenschweiler': 'HATTEN.TTF',
    'Magneto': 'MAGNETOB.TTF',
    'MV Boli': 'mvboli.ttf',
    'Curlz MT': 'CURLZ___.TTF',
    'Javanese Text': 'javatext.ttf',
    'FrankRuehl': 'frank.ttf',
    'Onyx': 'ONYX.TTF',
    'Poor Richard': 'POORICH.TTF',
    'Gadugi': 'gadugib.ttf',
    'Papyrus': 'PAPYRUS.TTF',
    'MS Gothic': 'msgothic.ttc'
}