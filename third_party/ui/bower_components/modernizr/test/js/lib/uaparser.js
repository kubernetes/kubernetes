// uaparser by lindsey simon,
// ported to node by tobie
// https://github.com/tobie/ua-parser/

// browserized by paul irish

(function(exports){

  exports.uaparse = parse;
  
  function parse(ua) {
    for (var i=0; i < parsers.length; i++) {
      var result = parsers[i](ua);
      if (result) { return result; }
    }
    return new UserAgent();
  }

  function UserAgent(family) {
    this.family = family || 'Other';
  }

  UserAgent.prototype.toVersionString = function() {
    var output = '';
    if (this.major != null) {
      output += this.major;
      if (this.minor != null) {
        output += '.' + this.minor;
        if (this.patch != null) {
          output += '.' + this.patch;
        }
      }
    }
    return output;
  };

  UserAgent.prototype.toString = function() {
    var suffix = this.toVersionString();
    if (suffix) { suffix = ' ' + suffix; }
    return this.family + suffix;
  };
  
  
  var regexes = [
      {"pattern":"^(Opera)/(\\d+)\\.(\\d+) \\(Nintendo Wii",
       "v1_replacement":null,
       "family_replacement":"Wii"},
      {"pattern":"(Namoroka|Shiretoko|Minefield)/(\\d+)\\.(\\d+)\\.(\\d+(?:pre)?)",
       "v1_replacement":null,
       "family_replacement":"Firefox ($1)"},
      {"pattern":"(Namoroka|Shiretoko|Minefield)/(\\d+)\\.(\\d+)([ab]\\d+[a-z]*)?",
       "v1_replacement":null,
       "family_replacement":"Firefox ($1)"},
      {"pattern":"(SeaMonkey|Fennec|Camino)/(\\d+)\\.(\\d+)([ab]?\\d+[a-z]*)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Flock)/(\\d+)\\.(\\d+)(b\\d+?)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Fennec)/(\\d+)\\.(\\d+)(pre)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Navigator)/(\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Netscape"},
      {"pattern":"(Navigator)/(\\d+)\\.(\\d+)([ab]\\d+)",
       "v1_replacement":null,
       "family_replacement":"Netscape"},
      {"pattern":"(Netscape6)/(\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Netscape"},
      {"pattern":"(MyIBrow)/(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"My Internet Browser"},
      {"pattern":"(Firefox).*Tablet browser (\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"MicroB"},
      {"pattern":"(Opera)/9.80.*Version\\/(\\d+)\\.(\\d+)(?:\\.(\\d+))?",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Firefox)/(\\d+)\\.(\\d+)\\.(\\d+(?:pre)?) \\(Swiftfox\\)",
       "v1_replacement":null,
       "family_replacement":"Swiftfox"},
      {"pattern":"(Firefox)/(\\d+)\\.(\\d+)([ab]\\d+[a-z]*)? \\(Swiftfox\\)",
       "v1_replacement":null,
       "family_replacement":"Swiftfox"},
      {"pattern":"(konqueror)/(\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Konqueror"},
      {"pattern":"(Jasmine|ANTGalio|Midori|Fresco|Lobo|Maxthon|Lynx|OmniWeb|Dillo|Camino|Demeter|Fluid|Fennec|Shiira|Sunrise|Chrome|Flock|Netscape|Lunascape|Epiphany|WebPilot|Vodafone|NetFront|Konqueror|SeaMonkey|Kazehakase|Vienna|Iceape|Iceweasel|IceWeasel|Iron|K-Meleon|Sleipnir|Galeon|GranParadiso|Opera Mini|iCab|NetNewsWire|Iron|Iris)/(\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Bolt|Jasmine|Maxthon|Lynx|Arora|IBrowse|Dillo|Camino|Shiira|Fennec|Phoenix|Chrome|Flock|Netscape|Lunascape|Epiphany|WebPilot|Opera Mini|Opera|Vodafone|NetFront|Konqueror|SeaMonkey|Kazehakase|Vienna|Iceape|Iceweasel|IceWeasel|Iron|K-Meleon|Sleipnir|Galeon|GranParadiso|iCab|NetNewsWire|Iron|Space Bison|Stainless|Orca)/(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(iRider|Crazy Browser|SkipStone|iCab|Lunascape|Sleipnir|Maemo Browser) (\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(iCab|Lunascape|Opera|Android) (\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(IEMobile) (\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"IE Mobile"},
      {"pattern":"(Firefox)/(\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Firefox)/(\\d+)\\.(\\d+)(pre|[ab]\\d+[a-z]*)?",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Obigo|OBIGO)[^\\d]*(\\d+)(?:.(\\d+))?",
       "v1_replacement":null,
       "family_replacement":"Obigo"},
      {"pattern":"(MAXTHON|Maxthon) (\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Maxthon"},
      {"pattern":"(Maxthon|MyIE2|Uzbl|Shiira)",
       "v1_replacement":"0",
       "family_replacement":null},
      {"pattern":"(PLAYSTATION) (\\d+)",
       "v1_replacement":null,
       "family_replacement":"PlayStation"},
      {"pattern":"(PlayStation Portable)[^\\d]+(\\d+).(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(BrowseX) \\((\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Opera)/(\\d+)\\.(\\d+).*Opera Mobi",
       "v1_replacement":null,
       "family_replacement":"Opera Mobile"},
      {"pattern":"(POLARIS)/(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Polaris"},
      {"pattern":"(BonEcho)/(\\d+)\\.(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Bon Echo"},
      {"pattern":"(iPhone) OS (\\d+)_(\\d+)(?:_(\\d+))?",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Avant)",
       "v1_replacement":"1",
       "family_replacement":null},
      {"pattern":"(Nokia)[EN]?(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Black[bB]erry)(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Blackberry"},
      {"pattern":"(OmniWeb)/v(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Blazer)/(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Palm Blazer"},
      {"pattern":"(Pre)/(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"Palm Pre"},
      {"pattern":"(Links) \\((\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(QtWeb) Internet Browser/(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(Version)/(\\d+)\\.(\\d+)(?:\\.(\\d+))?.*Safari/",
       "v1_replacement":null,
       "family_replacement":"Safari"},
      {"pattern":"(OLPC)/Update(\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(OLPC)/Update()\\.(\\d+)",
       "v1_replacement":"0",
       "family_replacement":null},
      {"pattern":"(SamsungSGHi560)",
       "v1_replacement":null,
       "family_replacement":"Samsung SGHi560"},
      {"pattern":"^(SonyEricssonK800i)",
       "v1_replacement":null,
       "family_replacement":"Sony Ericsson K800i"},
      {"pattern":"(Teleca Q7)",
       "v1_replacement":null,
       "family_replacement":null},
      {"pattern":"(MSIE) (\\d+)\\.(\\d+)",
       "v1_replacement":null,
       "family_replacement":"IE"}

  ];
  
  var parsers = regexes.map(function(obj) {
    var regexp = new RegExp(obj.pattern),
        famRep = obj.family_replacement,
        v1Rep = obj.v1_replacement;

    function parser(ua) {
      var m = ua.match(regexp);

      if (!m) { return null; }

      var familly = famRep ? famRep.replace('$1', m[1]) : m[1];

      var obj = new UserAgent(familly);
      obj.major = parseInt(v1Rep ? v1Rep : m[2]);
      obj.minor = m[3] ? parseInt(m[3]) : null;
      obj.patch = m[4] ? parseInt(m[4]) : null;

      return obj;
    }

    return parser;
  });
  
  
})(window);


