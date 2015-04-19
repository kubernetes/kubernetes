if (typeof exports !== 'undefined') {
	var window = {Unicode: require('./unicodecategories').Unicode};
	exports.Tokenizer = Tokenizer;
}

/*!
 * Tokenizer for JavaScript / ECMAScript 5
 * (c) Peter van der Zee, qfox.nl
 */

/**
 * @param {Object} inp
 */
function Tokenizer(inp){
	this.inp = inp||'';
	// replace all other line terminators with \n (leave \r\n in tact though). we should probably remove the shadowInp when finished...
	// only replace \r if it is not followed by a \n else \r\n would become \n\n causing a double newline where it is just a single
	this.shadowInp = (inp||'').replace(Tokenizer.regexNormalizeNewlines, '\n');
	this.pos = 0;
	this.line = 0;
	this.column = 0;
	this.cache = {};
	
	this.errorStack = [];
	
	this.wtree = [];
	this.btree = [];
	
//	this.regexWhiteSpace = Tokenizer.regexWhiteSpace;
	this.regexLineTerminator = Tokenizer.regexLineTerminator; // used in fallback
	this.regexAsciiIdentifier = Tokenizer.regexAsciiIdentifier;
	this.hashAsciiIdentifier = Tokenizer.hashAsciiIdentifier;
//	this.regexHex = Tokenizer.regexHex;
	this.hashHex = Tokenizer.hashHex
	this.regexUnicodeEscape = Tokenizer.regexUnicodeEscape;
	this.regexIdentifierStop = Tokenizer.regexIdentifierStop;
	this.hashIdentifierStop = Tokenizer.hashIdentifierStop;
//	this.regexPunctuators = Tokenizer.regexPunctuators;
	this.regexNumber = Tokenizer.regexNumber;
	this.regexNewline = Tokenizer.regexNewline;
	
	this.regexBig = Tokenizer.regexBig;
	this.regexBigAlt = Tokenizer.regexBigAlt;
	
	this.tokenCount = 0;
	this.tokenCountNoWhite = 0;
	
	this.Unicode = window.Unicode;
	
	// if the Parser throws an error. it will set this property to the next match
	// at the time of the error (which was not what it was expecting at that point) 
	// and pass on an "error" match. the error should be scooped on the stack and 
	// this property should be returned, without looking at the input...
	this.errorEscape = null;
};

Tokenizer.prototype = {
	inp:null,
	shadowInp:null,
	pos:null,
	line:null,
	column:null,
	cache:null,
	errorStack:null,
	
	wtree: null, // contains whitespace (spaces, comments, newlines)
	btree: null, // does not contain any whitespace tokens.
	
	regexLineTerminator:null,
	regexAsciiIdentifier:null,
	hashAsciiIdentifier:null,
	hashHex:null,
	regexUnicodeEscape:null,
	regexIdentifierStop:null,
	hashIdentifierStop:null,
	regexNumber:null,
	regexNewline:null,
	regexBig:null,
	regexBigAlt:null,
	tokenCount:null,
	tokenCountNoWhite:null,
	
	Unicode:null,
	
	// storeCurrentAndFetchNextToken(bool, false, false true) to get just one token
	storeCurrentAndFetchNextToken: function(noRegex, returnValue, stack, _dontStore){
		var regex = !noRegex; // TOFIX :)
		var pos = this.pos;
		var inp = this.inp;
		var shadowInp = this.shadowInp;
		var matchedNewline = false;
		do {
			if (!_dontStore) {
				++this.tokenCount;
				stack.push(returnValue);
				// did the parent Parser throw up?
				if (this.errorEscape) {
					returnValue = this.errorEscape;
					this.errorEscape = null;
					return returnValue;
				}
			}
			_dontStore = false;
		
			if (pos >= inp.length) {
				returnValue = {start:inp.length,stop:inp.length,name:12/*EOF*/};
				break; 
			}
			var returnValue = null;
		
			var start = pos;
			var chr = inp[pos];
	
			//							1 ws							2 lt				   3 scmt 4 mcmt 5/6 str 7 nr     8 rx  9 punc
			//if (true) {
				// substring method (I think this is faster..)
				var part2 = inp.substring(pos,pos+4);
				var part = this.regexBig.exec(part2);
			//} else {
			//	// non-substring method (lastIndex)
			//	// this method does not need a substring to apply it
			//	this.regexBigAlt.lastIndex = pos;
			//	var part = this.regexBigAlt.exec(inp);
			//}
			
			if (part[1]) { //this.regexWhiteSpace.test(chr)) { // SP, TAB, VT, FF, NBSP, BOM (, TOFIX: USP)
				++pos;
				returnValue = {start:start,stop:pos,name:9/*WHITE_SPACE*/,line:this.line,col:this.column,isWhite:true};
				++this.column;
			} else if (part[2]) { //this.regexLineTerminator.test(chr)) { // LF, CR, LS, PS
				var end = pos+1;
				if (chr=='\r' && inp[pos+1] == '\n') ++end; // support crlf=>lf
				returnValue = {start:pos,stop:end,name:10/*LINETERMINATOR*/,line:this.line,col:this.column,isWhite:true};
				pos = end;
				// mark newlines for ASI
				matchedNewline = true;
				++this.line;
				this.column = 0;
				returnValue.hasNewline = 1;
			} else if (part[3]) { //chr == '/' && inp[pos+1] == '/') {
				pos = shadowInp.indexOf('\n',pos);
				if (pos == -1) pos = inp.length;
				returnValue = {start:start,stop:pos,name:7/*COMMENT_SINGLE*/,line:this.line,col:this.column,isComment:true,isWhite:true};
				this.column = returnValue.stop;
			} else if (part[4]) { //chr == '/' && inp[pos+1] == '*') {
				var newpos = inp.indexOf('*/',pos);
				if (newpos == -1) {
					newpos = shadowInp.indexOf('\n', pos);
					if (newpos < 0) pos += 2;
					else pos = newpos;
					returnValue = {start:start,stop:pos,name:14/*error*/,value:inp.substring(start, pos),line:this.line,col:this.column,isComment:true,isWhite:true,tokenError:true,error:Tokenizer.Error.UnterminatedMultiLineComment};
					this.errorStack.push(returnValue);
				} else {
					pos = newpos+2;
					returnValue = {start:start,stop:pos,name:8/*COMMENT_MULTI*/,value:inp.substring(start, pos),line:this.line,col:this.column,isComment:true,isWhite:true};
	
					// multi line comments are also reason for asi, but only if they contain at least one newline (use shadow input, because all line terminators would be valid...)
					var shadowValue = shadowInp.substring(start, pos);
					var i = 0, hasNewline = 0;
					while (i < (i = shadowValue.indexOf('\n', i+1))) {
						++hasNewline;
					}
					if (hasNewline) {
						matchedNewline = true;
						returnValue.hasNewline = hasNewline;
						this.line += hasNewline;
						this.column = 0;
					} else {
						this.column = returnValue.stop;
					}
				}
			} else if (part[5]) { //chr == "'") {
				// old method
				//console.log("old method");
				
				var hasNewline = 0;
				do {
					// process escaped characters
					while (pos < inp.length && inp[++pos] == '\\') {
						if (shadowInp[pos+1] == '\n') ++hasNewline;
						++pos;
					}
					if (this.regexLineTerminator.test(inp[pos])) {
						returnValue = {start:start,stop:pos,name:14/*error*/,value:inp.substring(start, pos),isString:true,tokenError:true,error:Tokenizer.Error.UnterminatedDoubleStringNewline};
						this.errorStack.push(returnValue);
						break;
					}
				} while (pos < inp.length && inp[pos] != "'");
				if (returnValue) {} // error
				else if (inp[pos] != "'") {
					returnValue = {start:start,stop:pos,name:14/*error*/,value:inp.substring(start, pos),isString:true,tokenError:true,error:Tokenizer.Error.UnterminatedDoubleStringOther};
					this.errorStack.push(returnValue);
				} else {
					++pos;
					returnValue = {start:start,stop:pos,name:5/*STRING_SINGLE*/,isPrimitive:true,isString:true};
					if (hasNewline) {
						returnValue.hasNewline = hasNewline;
						this.line += hasNewline;
						this.column = 0;
					} else {
						this.column += (pos-start);
					}
				}				
			} else if (part[6]) { //chr == '"') {
				var hasNewline = 0;
				// TODO: something like this: var regexmatch = /([^\']|$)+/.match();
				do {
					// process escaped chars
					while (pos < inp.length && inp[++pos] == '\\') {
						if (shadowInp[pos+1] == '\n') ++hasNewline;
						++pos;
					}
					if (this.regexLineTerminator.test(inp[pos])) {
						returnValue = {start:start,stop:pos,name:14/*error*/,value:inp.substring(start, pos),isString:true,tokenError:true,error:Tokenizer.Error.UnterminatedSingleStringNewline};
						this.errorStack.push(returnValue);
						break;
					}
				} while (pos < inp.length && inp[pos] != '"');
				if (returnValue) {}
				else if (inp[pos] != '"') {
					returnValue = {start:start,stop:pos,name:14/*error*/,value:inp.substring(start, pos),isString:true,tokenError:true,error:Tokenizer.Error.UnterminatedSingleStringOther};
					this.errorStack.push(returnValue);
				} else {
					++pos;
					returnValue = {start:start,stop:pos,name:6/*STRING_DOUBLE*/,isPrimitive:true,isString:true};
					if (hasNewline) {
						returnValue.hasNewline = hasNewline;
						this.line += hasNewline;
						this.column = 0;
					} else {
						this.column += (pos-start);
					}
				}
			} else if (part[7]) { //(chr >= '0' && chr <= '9') || (chr == '.' && inp[pos+1] >= '0' && inp[pos+1] <= '9')) {
				var nextPart = inp.substring(pos, pos+30);
				var match = nextPart.match(this.regexNumber);
				if (match[2]) { // decimal
					var value = match[2];
					var parsingOctal = value[0] == '0' && value[1] && value[1] != 'e' && value[1] != 'E' && value[1] != '.';
					if (parsingOctal) {
						returnValue = {start:start,stop:pos,name:14/*error*/,isNumber:true,isOctal:true,tokenError:true,error:Tokenizer.Error.IllegalOctalEscape,value:value};
						this.errorStack.push(returnValue);
					} else {
						returnValue = {start:start,stop:start+value.length,name:4/*NUMERIC_DEC*/,isPrimitive:true,isNumber:true,value:value};
					}
				} else if (match[1]) { // hex
					var value = match[1];
					returnValue = {start:start,stop:start+value.length,name:3/*NUMERIC_HEX*/,isPrimitive:true,isNumber:true,value:value};
				} else {
					throw 'unexpected parser errror... regex fail :(';
				}
				
				if (value.length < 300) {
					pos += value.length;
				} else {
					// old method of parsing numbers. only used for extremely long number literals (300+ chars).
					// this method does not require substringing... just memory :)
					var tmpReturnValue = this.oldNumberParser(pos, chr, inp, returnValue, start, Tokenizer);
					pos = tmpReturnValue[0];
					returnValue = tmpReturnValue[1];
				}
			} else if (regex && part[8]) { //chr == '/') { // regex cannot start with /* (would be multiline comment, and not make sense anyways). but if it was /* then an earlier if would have eated it. so we only check for /
				var twinfo = []; // matching {[( info
				var found = false;
				var parens = [];
				var nonLethalError = null;
				while (++pos < inp.length) {
					chr = shadowInp[pos];
					// parse RegularExpressionChar
					if (chr == '\n') {
						returnValue = {start:start,stop:pos,name:14/*error*/,tokenError:true,errorHasContent:true,error:Tokenizer.Error.UnterminatedRegularExpressionNewline};
						this.errorStack.push(returnValue);
						break; // fail
					} else if (chr == '/') {
						found = true;
						break;
					} else if (chr == '?' || chr == '*' || chr == '+') {
						nonLethalError = Tokenizer.Error.NothingToRepeat;
					} else if (chr == '^') {
						if (
							inp[pos-1] != '/' && 
							inp[pos-1] != '|' && 
							inp[pos-1] != '(' &&
							!(inp[pos-3] == '(' && inp[pos-2] == '?' && (inp[pos-1] == ':' || inp[pos-1] == '!' || inp[pos-1] == '='))
						) {
							nonLethalError = Tokenizer.Error.StartOfMatchShouldBeAtStart;
						}
					} else if (chr == '$') {
						if (inp[pos+1] != '/' && inp[pos+1] != '|' && inp[pos+1] != ')') nonLethalError = Tokenizer.Error.DollarShouldBeEnd;
					} else if (chr == '}') {
						nonLethalError = Tokenizer.Error.MissingOpeningCurly;
					} else { // it's a "character" (can be group or class), something to match
						// match parenthesis
						if (chr == '(') {
							parens.push(pos-start);
						} else if (chr == ')') {
							if (parens.length == 0) {
								nonLethalError = {start:start,stop:pos,name:14/*error*/,tokenError:true,error:Tokenizer.Error.RegexNoOpenGroups};
							} else {
								var twin = parens.pop();
								var now = pos-start;
								twinfo[twin] = now;
								twinfo[now] = twin;
							}
						}
						// first process character class
						if (chr == '[') {
							var before = pos-start;
							while (++pos < inp.length && shadowInp[pos] != '\n' && inp[pos] != ']') {
								// only newline is not allowed in class range
								// anything else can be escaped, most of it does not have to be escaped...
								if (inp[pos] == '\\') {
									if (shadowInp[pos+1] == '\n') break;
									else ++pos; // skip next char. (mainly prohibits ] to be picked up as closing the group...)
								}
							} 
							if (inp[pos] != ']') {
								returnValue = {start:start,stop:pos,name:14/*error*/,tokenError:true,error:Tokenizer.Error.ClosingClassRangeNotFound};
								this.errorStack.push(returnValue);
								break;
							} else {
								var after = pos-start;
								twinfo[before] = after;
								twinfo[after] = before;
							}
						} else if (chr == '\\' && shadowInp[pos+1] != '\n') {
							// is ok anywhere in the regex (match next char literally, regardless of its otherwise special meaning)
							++pos;
						}
						
						// now process repeaters (+, ? and *)
						
						// non-collecting group (?:...) and positive (?=...) or negative (?!...) lookahead
						if (chr == '(') {
							if (inp[pos+1] == '?' && (inp[pos+2] == ':' || inp[pos+2] == '=' || inp[pos+2] == '!')) {
								pos += 2;
							}
						}
						// matching "char"
						else if (inp[pos+1] == '?') ++pos;
						else if (inp[pos+1] == '*' || inp[pos+1] == '+') {
							++pos;
							if (inp[pos+1] == '?') ++pos; // non-greedy match
						} else if (inp[pos+1] == '{') {
							pos += 1;
							var before = pos-start;
							// quantifier:
							// - {n}
							// - {n,}
							// - {n,m}
							if (!/[0-9]/.test(inp[pos+1])) {
								nonLethalError = Tokenizer.Error.QuantifierRequiresNumber;
							}
							while (++pos < inp.length && /[0-9]/.test(inp[pos+1]));
							if (inp[pos+1] == ',') {
								++pos;
								while (pos < inp.length && /[0-9]/.test(inp[pos+1])) ++pos;
							}
							if (inp[pos+1] != '}') {
								nonLethalError = Tokenizer.Error.QuantifierRequiresClosingCurly;
							} else {
								++pos;
								var after = pos-start;
								twinfo[before] = after;
								twinfo[after] = before;
								if (inp[pos+1] == '?') ++pos; // non-greedy match
							}
						}
					}
				}
				// if found=false, fail right now. otherwise try to parse an identifiername (that's all RegularExpressionFlags is..., but it's constructed in a stupid fashion)
				if (!found || returnValue) {
					if (!returnValue) {
						returnValue = {start:start,stop:pos,name:14/*error*/,tokenError:true,error:Tokenizer.Error.UnterminatedRegularExpressionOther};
						this.errorStack.push(returnValue);
					}
				} else {
					// this is the identifier scanner, for now
					do ++pos;
					while (pos < inp.length && this.hashAsciiIdentifier[inp[pos]]); /*this.regexAsciiIdentifier.test(inp[pos])*/ 
	
					if (parens.length) {
						// nope, this is still an error, there was at least one paren that did not have a matching twin
						if (parens.length > 0) returnValue = {start:start,stop:pos,name:14/*error*/,tokenError:true,error:Tokenizer.Error.RegexOpenGroup};
						this.errorStack.push(returnValue);
					} else if (nonLethalError) {
						returnValue = {start:start,stop:pos,name:14/*error*/,errorHasContent:true,tokenError:true,error:nonLethalError};
						this.errorStack.push(returnValue);
					} else {
						returnValue = {start:start,stop:pos,name:1/*REG_EX*/,isPrimitive:true};
					}				
				}
				returnValue.twinfo = twinfo;
			} else {
				// note: operators need to be ordered from longest to smallest. regex will take care of the rest.
				// no need to worry about div vs regex. if looking for regex, earlier if will have eaten it
				//var result = this.regexPunctuators.exec(inp.substring(pos,pos+4));
				
				// note: due to the regex, the single forward slash might be caught by an earlier part of the regex. so check for that.
				var result = part[8] || part[9];
				if (result) {
					//result = result[1];
					returnValue = {start:pos,stop:pos+=result.length,name:11/*PUNCTUATOR*/,value:result};
				} else {
					var found = false;
					// identifiers cannot start with a number. but if the leading string would be a number, another if would have eaten it already for numeric literal :)
					while (pos < inp.length) {
						var c = inp[pos];
	
						if (this.hashAsciiIdentifier[c]) ++pos; //if (this.regexAsciiIdentifier.test(c)) ++pos;
						else if (c == '\\' && this.regexUnicodeEscape.test(inp.substring(pos,pos+6))) pos += 6; // this is like a \uxxxx
						// ok, now test unicode ranges...
						// basically this hardly ever happens so there's little risk of this hitting performance
						// however, if you do happen to have used them, it's not a problem. the parser will support it :)
						else if (this.Unicode) { // the unicode is optional.
							// these chars may not be part of identifier. i want to try to prevent running the unicode regexes here...
							if (this.hashIdentifierStop[c] /*this.regexIdentifierStop.test(c)*/) break;
							// for most scripts, the code wont reach here. which is good, because this is going to be relatively slow :)
							var Unicode = this.Unicode; // cache
							if (!(
									// these may all occur in an identifier... (pure a specification compliance thing :)
									Unicode.Lu.test(c) || Unicode.Ll.test(c) || Unicode.Lt.test(c) || Unicode.Lm.test(c) || 
									Unicode.Lo.test(c) || Unicode.Nl.test(c) || Unicode.Mn.test(c) || Unicode.Mc.test(c) ||
									Unicode.Nd.test(c) || Unicode.Pc.test(c) || Unicode.sp.test(c)
							)) break; // end of match.
							// passed, next char
							++pos;
						} else break; // end of match.
			
						found = true;
					}
		
					if (found) {
						returnValue = {start:start,stop:pos,name:2/*IDENTIFIER*/,value:inp.substring(start,pos)};
						if (returnValue.value == 'undefined' || returnValue.value == 'null' || returnValue.value == 'true' || returnValue.value == 'false') returnValue.isPrimitive = true;
					} else {
						if (inp[pos] == '`') {
							returnValue = {start:start,stop:pos+1,name:14/*error*/,tokenError:true,error:Tokenizer.Error.BacktickNotSupported};
							this.errorStack.push(returnValue);
						} else if (inp[pos] == '\\') {
							if (inp[pos+1] == 'u') {
								returnValue = {start:start,stop:pos+1,name:14/*error*/,tokenError:true,error:Tokenizer.Error.InvalidUnicodeEscape};
								this.errorStack.push(returnValue);
							} else {
								returnValue = {start:start,stop:pos+1,name:14/*error*/,tokenError:true,error:Tokenizer.Error.InvalidBackslash};
								this.errorStack.push(returnValue);
							}
						} else {
							returnValue = {start:start,stop:pos+1,name:14/*error*/,tokenError:true,error:Tokenizer.Error.Unknown,value:c};
							this.errorStack.push(returnValue);
							// try to skip this char. it's not going anywhere.
						}
						++pos;
					}
				}
			}
			
			if (returnValue) {
				// note that ASI's are slipstreamed in here from the parser since the tokenizer cant determine that
				// if this part ever changes, make sure you change that too :)
				returnValue.tokposw = this.wtree.length;
				this.wtree.push(returnValue);
				if (!returnValue.isWhite) {
					returnValue.tokposb = this.btree.length;
					this.btree.push(returnValue);
				} 
			}
			
			
		} while (stack && returnValue && returnValue.isWhite); // WHITE_SPACE LINETERMINATOR COMMENT_SINGLE COMMENT_MULTI
		++this.tokenCountNoWhite;
		
		this.pos = pos;
	
		if (matchedNewline) returnValue.newline = true;
		return returnValue;
	},
	addTokenToStreamBefore: function(token, match){
		var wtree = this.wtree;
		var btree = this.btree;
		if (match.name == 12/*asi*/) {
			token.tokposw = wtree.length;
			wtree.push(token);
			token.tokposb = btree.length;
			btree.push(token);
		} else {
			token.tokposw = match.tokposw;
			wtree[token.tokposw] = token;
			match.tokposw += 1;
			wtree[match.tokposw] = match;

			if (match.tokposb) {
				token.tokposb = match.tokposb;
				btree[token.tokposb] = token;
				match.tokposb += 1;
				btree[match.tokposb] = match;
			}
		}
	},
	oldNumberParser: function(pos, chr, inp, returnValue, start, Tokenizer){
		++pos;
		// either: 0x 0X 0 .3
		if (chr == '0' && (inp[pos] == 'x' || inp[pos] == 'X')) {
			// parsing hex
			while (++pos < inp.length && this.hashHex[inp[pos]]); // this.regexHex.test(inp[pos]));
			returnValue = {start:start,stop:pos,name:3/*NUMERIC_HEX*/,isPrimitive:true,isNumber:true};
		} else {
			var parsingOctal = chr == '0' && inp[pos] >= '0' && inp[pos] <= '9';
			// parsing dec
			if (chr != '.') { // integer part
				while (pos < inp.length && inp[pos] >= '0' && inp[pos] <= '9') ++pos;
				if (inp[pos] == '.') ++pos;
			}
			// decimal part
			while (pos < inp.length && inp[pos] >= '0' && inp[pos] <= '9') ++pos;
			// exponent part
			if (inp[pos] == 'e' || inp[pos] == 'E') {
				if (inp[++pos] == '+' || inp[pos] == '-') ++pos;
				var expPosBak = pos;
				while (pos < inp.length && inp[pos] >= '0' && inp[pos] <= '9') ++pos;
				if (expPosBak == pos) {
					returnValue = {start:start,stop:pos,name:14/*error*/,tokenError:true,error:Tokenizer.Error.NumberExponentRequiresDigits};
					this.errorStack.push(returnValue);
				}
			}
			if (returnValue.name != 14/*error*/) {
				if (parsingOctal) {
					returnValue = {start:start,stop:pos,name:14/*error*/,isNumber:true,isOctal:true,tokenError:true,error:Tokenizer.Error.IllegalOctalEscape};
					this.errorStack.push(returnValue);
					console.log("foo")
				} else {
					returnValue = {start:start,stop:pos,name:4/*NUMERIC_DEC*/,isPrimitive:true,isNumber:true};
				}
			}
		}
		return [pos, returnValue];
	},
	tokens: function(arrx){
		arrx = arrx || [];
		var n = 0;
		var last;
		var stack = [];
		while ((last = this.storeCurrentAndFetchNextToken(!arrx[n++], false, false, true)) && last.name != 12/*EOF*/) stack.push(last);
		return stack;
	},
	fixValues: function(){
		this.wtree.forEach(function(t){
			if (!t.value) t.value = this.inp.substring(t.start, t.stop);
		},this);
	}
};

//#ifdef TEST_SUITE
Tokenizer.escape = function(s){
	return s.replace(/\n/g,'\\n').replace(/\t/g,'\\t').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\uFFFF/g, '\\uFFFF').replace(/\s/g, function(s){
		// replace whitespace as is...
		var ord = s.charCodeAt(0).toString(16);
		switch (ord.length) {
			case 1: ord = '000'+ord; break;
			case 2: ord = '00'+ord; break;
			case 3: ord = '0'+ord; break;
		}
		return '\\u'+ord;
	});
};
Tokenizer.testSuite = function(arr){
	var out = document.createElement('pre');
	document.body.appendChild(out);
	var debug = function(){
		var f = document.createElement('div');
		f.innerHTML = Array.prototype.slice.call(arguments).join(' ');
		out.appendChild(f);
		return arguments[0];
	};

	debug("Running test suite...",arr.length,"tests");
	debug(' ');
	var start = +new Date;
	var ok = 0;
	var fail = 0;
	for (var i=0; i<arr.length; ++i) {
		var test = arr[i], result;
		var input = test[1];
		var outputLen = test[2];
		var regexHints = test[4] ? test[3] : null; // if flags, then len=4
		var desc = test[4] || test[3];
		
		var result = new Tokenizer(input).tokens(regexHints); // regexHints can be null, that's ok
		if (result.length == outputLen) {
			debug('<span class="green">Test '+i+' ok:</span>',desc);
			++ok;
		} else {
			debug('<b class="red">Test failed:</span>',desc,'(found',result.length,'expected',outputLen+')'),console.log(desc, result);
			++fail;
		}
		debug('<b>'+Tokenizer.escape(input)+'</b>');
		debug('<br/>');
	}
	debug("Tokenizer test suite finished ("+(+new Date - start)+' ms). ok:'+ok+', fail:'+fail);
};
//#endif

Tokenizer.regexWhiteSpace = /[ \t\u000B\u000C\u00A0\uFFFF]/;
Tokenizer.regexLineTerminator = /[\u000A\u000D\u2028\u2029]/;
Tokenizer.regexAsciiIdentifier = /[a-zA-Z0-9\$_]/;
Tokenizer.hashAsciiIdentifier = {_:1,$:1,a:1,b:1,c:1,d:1,e:1,f:1,g:1,h:1,i:1,j:1,k:1,l:1,m:1,n:1,o:1,p:1,q:1,r:1,s:1,t:1,u:1,v:1,w:1,x:1,y:1,z:1,A:1,B:1,C:1,D:1,E:1,F:1,G:1,H:1,I:1,J:1,K:1,L:1,M:1,N:1,O:1,P:1,Q:1,R:1,S:1,T:1,U:1,V:1,W:1,X:1,Y:1,Z:1,0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1};
Tokenizer.regexHex = /[0-9A-Fa-f]/;
Tokenizer.hashHex = {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,a:1,b:1,c:1,d:1,e:1,f:1,A:1,B:1,C:1,D:1,E:1,F:1};
Tokenizer.regexUnicodeEscape = /u[0-9A-Fa-f]{4}/; // the \ is already checked at usage...
Tokenizer.regexIdentifierStop = /[\>\=\!\|\<\+\-\&\*\%\^\/\{\}\(\)\[\]\.\;\,\~\?\:\ \t\n\\\'\"]/; 
Tokenizer.hashIdentifierStop = {'>':1,'=':1,'!':1,'|':1,'<':1,'+':1,'-':1,'&':1,'*':1,'%':1,'^':1,'/':1,'{':1,'}':1,'(':1,')':1,'[':1,']':1,'.':1,';':1,',':1,'~':1,'?':1,':':1,'\\':1,'\'':1,'"':1,' ':1,'\t':1,'\n':1};
Tokenizer.regexNewline = /\n/g;
//Tokenizer.regexPunctuators = /^(>>>=|===|!==|>>>|<<=|>>=|<=|>=|==|!=|\+\+|--|<<|>>|\&\&|\|\||\+=|-=|\*=|%=|\&=|\|=|\^=|\/=|\{|\}|\(|\)|\[|\]|\.|;|,|<|>|\+|-|\*|%|\||\&|\||\^|!|~|\?|:|=|\/)/;
Tokenizer.Unidocde = window.Unicode;
Tokenizer.regexNumber = /^(?:(0[xX][0-9A-Fa-f]+)|((?:(?:(?:(?:[0-9]+)(?:\.[0-9]*)?))|(?:\.[0-9]+))(?:[eE][-+]?[0-9]{1,})?))/;
Tokenizer.regexNormalizeNewlines = /(\u000D[^\u000A])|[\u2028\u2029]/;

//							1 ws							2 lt				   3 scmt 4 mcmt 5/6 str 7 nr     8 rx  9 punc
Tokenizer.regexBig = /^([ \t\u000B\u000C\u00A0\uFFFF])?([\u000A\u000D\u2028\u2029])?(\/\/)?(\/\*)?(')?(")?(\.?[0-9])?(?:(\/)[^=])?(>>>=|===|!==|>>>|<<=|>>=|<=|>=|==|!=|\+\+|--|<<|>>|\&\&|\|\||\+=|-=|\*=|%=|\&=|\|=|\^=|\/=|\{|\}|\(|\)|\[|\]|\.|;|,|<|>|\+|-|\*|%|\||\&|\||\^|!|~|\?|:|=|\/)?/;
Tokenizer.regexBigAlt = /([ \t\u000B\u000C\u00A0\uFFFF])?([\u000A\u000D\u2028\u2029])?(\/\/)?(\/\*)?(')?(")?(\.?[0-9])?(?:(\/)[^=])?(>>>=|===|!==|>>>|<<=|>>=|<=|>=|==|!=|\+\+|--|<<|>>|\&\&|\|\||\+=|-=|\*=|%=|\&=|\|=|\^=|\/=|\{|\}|\(|\)|\[|\]|\.|;|,|<|>|\+|-|\*|%|\||\&|\||\^|!|~|\?|:|=|\/)?/g;

Tokenizer.Error = {
	UnterminatedSingleStringNewline: {msg:'Newlines are not allowed in string literals'},
	UnterminatedSingleStringOther: {msg:'Unterminated single string'},
	UnterminatedDoubleStringNewline: {msg:'Newlines are not allowed in string literals'},
	UnterminatedDoubleStringOther: {msg:'Unterminated double string'},
	UnterminatedRegularExpressionNewline: {msg:'Newlines are not allowed in regular expressions'},
	NothingToRepeat: {msg:'Used a repeat character (*?+) in a regex without something prior to it to match'},
	ClosingClassRangeNotFound: {msg: 'Unable to find ] for class range'},
	RegexOpenGroup: {msg: 'Open group did not find closing parenthesis'},
	RegexNoOpenGroups: {msg: 'Closing parenthesis found but no group open'},
	UnterminatedRegularExpressionOther: {msg:'Unterminated regular expression'},
	UnterminatedMultiLineComment: {msg:'Unterminated multi line comment'},
	UnexpectedIdentifier: {msg:'Unexpected identifier'},
	IllegalOctalEscape: {msg:'Octal escapes are not valid'},
	Unknown: {msg:'Unknown input'}, // if this happens, my parser is bad :(
	NumberExponentRequiresDigits: {msg:'Numbers with exponents require at least one digit after the `e`'},
	BacktickNotSupported: {msg:'The backtick is not used in js, maybe you copy/pasted from a fancy site/doc?'},
	InvalidUnicodeEscape: {msg:'Encountered an invalid unicode escape, must be followed by exactly four hex numbers'},
	InvalidBackslash: {msg:'Encountered a backslash where it not allowed'},
	StartOfMatchShouldBeAtStart: {msg: 'The ^ signifies the start of match but was not found at a start'},
	DollarShouldBeEnd: {msg: 'The $ signifies the stop of match but was not found at a stop'},
	QuantifierRequiresNumber: {msg:'Quantifier curly requires at least one digit before the comma'},
	QuantifierRequiresClosingCurly: {msg:'Quantifier curly requires to be closed'},
	MissingOpeningCurly: {msg:'Encountered closing quantifier curly without seeing an opening curly'}
};
