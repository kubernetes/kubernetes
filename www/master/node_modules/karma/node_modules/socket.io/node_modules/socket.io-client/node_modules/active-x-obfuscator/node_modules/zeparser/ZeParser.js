if (typeof exports !== 'undefined') {
	var Tokenizer = require('./Tokenizer').Tokenizer;
	exports.ZeParser = ZeParser;
}

/**
 * This is my js Parser: Ze. It's actually the post-dev pre-cleanup version. Clearly.
 * Some optimizations have been applied :)
 * (c) Peter van der Zee, qfox.nl
 * @param {String} inp Input
 * @param {Tokenizer} tok
 * @param {Array} stack The tokens will be put in this array. If you're looking for the AST, this would be it :)
 */
function ZeParser(inp, tok, stack, simple){
	this.input = inp;
	this.tokenizer = tok;
	this.stack = stack;
	this.stack.root = true;
	this.scope = stack.scope = [{value:'this', isDeclared:true, isEcma:true, thisIsGlobal:true}]; // names of variables
	this.scope.global = true;
	this.statementLabels = [];

	this.errorStack = [];

	stack.scope = this.scope; // hook root
	stack.labels = this.statementLabels;

	this.regexLhsStart = ZeParser.regexLhsStart;
/*
	this.regexStartKeyword = ZeParser.regexStartKeyword;
	this.regexKeyword = ZeParser.regexKeyword;
	this.regexStartReserved = ZeParser.regexStartReserved;
	this.regexReserved = ZeParser.regexReserved;
*/
	this.regexStartKeyOrReserved = ZeParser.regexStartKeyOrReserved;
	this.hashStartKeyOrReserved = ZeParser.hashStartKeyOrReserved;
	this.regexIsKeywordOrReserved = ZeParser.regexIsKeywordOrReserved;
	this.regexAssignments = ZeParser.regexAssignments;
	this.regexNonAssignmentBinaryExpressionOperators = ZeParser.regexNonAssignmentBinaryExpressionOperators;
	this.regexUnaryKeywords = ZeParser.regexUnaryKeywords;
	this.hashUnaryKeywordStart = ZeParser.hashUnaryKeywordStart;
	this.regexUnaryOperators = ZeParser.regexUnaryOperators;
	this.regexLiteralKeywords = ZeParser.regexLiteralKeywords;
	this.testing = {'this':1,'null':1,'true':1,'false':1};

	this.ast = !simple; ///#define FULL_AST
};
/**
 * Returns just a stacked parse tree (regular array)
 * @param {string} input
 * @param {boolean} simple=false
 * @return {Array}
 */
ZeParser.parse = function(input, simple){
	var tok = new Tokenizer(input);
	var stack = [];
	try {
		var parser = new ZeParser(input, tok, stack);
		if (simple) parser.ast = false;
		parser.parse();
		return stack;
	} catch (e) {
		console.log("Parser has a bug for this input, please report it :)", e);
		return null;
	}
};
/**
 * Returns a new parser instance with parse details for input
 * @param {string} input
 * @returns {ZeParser}
 */
ZeParser.createParser = function(input){
	var tok = new Tokenizer(input);
	var stack = [];
	try {
		var parser = new ZeParser(input, tok, stack);
		parser.parse();
		return parser;
	} catch (e) {
		console.log("Parser has a bug for this input, please report it :)", e);
		return null;
	}
};
ZeParser.prototype = {
	input: null,
	tokenizer: null,
	stack: null,
	scope: null,
	statementLabels: null,
	errorStack: null,

	ast: null,

	parse: function(match){
		if (match) match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, this.stack); // meh
		else match = this.tokenizer.storeCurrentAndFetchNextToken(false, null, this.stack, true); // initialization step, dont store the match (there isnt any!)

		match = this.eatSourceElements(match, this.stack);

		var cycled = false;
		do {
			if (match && match.name != 12/*eof*/) {
				// if not already an error, insert an error before it
				if (match.name != 14/*error*/) this.failignore('UnexpectedToken', match, this.stack);
				// just parse the token as is and continue.
				match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, this.stack);
				cycled = true;
			}

		// keep gobbling any errors...
		} while (match && match.name == 14/*error*/);

		// now try again (but only if we gobbled at least one token)...
		if (cycled && match && match.name != 12/*eof*/) match = this.parse(match);

		// pop the last token off the stack if it caused an error at eof
		if (this.tokenizer.errorEscape) {
			this.stack.push(this.tokenizer.errorEscape);
			this.tokenizer.errorEscape = null;
		}

		return match;
	},

	eatSemiColon: function(match, stack){
		//this.stats.eatSemiColon = (+//this.stats.eatSemiColon||0)+1;
		if (match.value == ';') match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		else {
			// try asi
			// only if:
			// - this token was preceeded by at least one newline (match.newline) or next token is }
			// - this is EOF
			// - prev token was one of return,continue,break,throw (restricted production), not checked here.

			// the exceptions to this rule are 
			// - if the next line is a regex 
			// - the semi is part of the for-header. 
			// these exceptions are automatically caught by the way the parser is built

			// not eof and just parsed semi or no newline preceeding and next isnt }
			if (match.name != 12/*EOF*/ && (match.semi || (!match.newline && match.value != '}')) && !(match.newline && (match.value == '++' || match.value == '--'))) {
				this.failignore('NoASI', match, stack);
			} else {
				// ASI
				// (match is actually the match _after_ this asi, so the position of asi is match.start, not stop (!)
				var asi = {start:match.start,stop:match.start,name:13/*ASI*/};
				stack.push(asi);
				
				// slip it in the stream, before the current match.
				// for the other tokens see the tokenizer near the end of the main parsing function
				this.tokenizer.addTokenToStreamBefore(asi, match);
			}
		}
		match.semi = true;
		return match;
	},
	/**
	 * Eat one or more "AssignmentExpression"s. May also eat a labeled statement if
	 * the parameters are set that way. This is the only way to linearly distinct between
	 * an expression-statement and a labeled-statement without double lookahead. (ok, maybe not "only")
	 * @param {boolean} mayParseLabeledStatementInstead=false If the first token is an identifier and the second a colon, accept this match as a labeled statement instead... Only true if the match in the parameter is an (unreserved) identifier (so no need to validate that further) 
	 * @param {Object} match
	 * @param {Array} stack
	 * @param {boolean} onlyOne=false Only parse a AssignmentExpression
	 * @param {boolean} forHeader=false Do not allow the `in` operator
	 * @param {boolean} isBreakOrContinueArg=false The argument for break or continue is always a single identifier
	 * @return {Object}
	 */
	eatExpressions: function(mayParseLabeledStatementInstead, match, stack, onlyOne, forHeader, isBreakOrContinueArg){
		if (this.ast) { //#ifdef FULL_AST
			var pstack = stack;
			stack = [];
			stack.desc = 'expressions';
			stack.nextBlack = match.tokposb;
			pstack.push(stack);

			var parsedExpressions = 0;
		} //#endif

		var first = true;
		do {
			var parsedNonAssignmentOperator = false; // once we parse a non-assignment, this expression can no longer parse an assignment
			// TOFIX: can probably get the regex out somehow...
			if (!first) {
				match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
				if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) match = this.failsafe('ExpectedAnotherExpressionComma', match);
			}

			if (this.ast) { //#ifdef FULL_AST
				++parsedExpressions;

				var astack = stack;
				stack = [];
				stack.desc = 'expression';
				stack.nextBlack = match.tokposb;
				astack.push(stack);
			} //#endif

			// start of expression is given: match
			// it should indeed be a properly allowed lhs
			// first eat all unary operators
			// they can be added to the stack, but we need to ensure they have indeed a valid operator

			var parseAnotherExpression = true;
			while (parseAnotherExpression) { // keep parsing lhs+operator as long as there is an operator after the lhs.
				if (this.ast) { //#ifdef FULL_AST
					var estack = stack;
					stack = [];
					stack.desc = 'sub-expression';
					stack.nextBlack = match.tokposb;
					estack.push(stack);

					var news = 0; // encountered new operators waiting for parenthesis
				} //#endif

				// start checking lhs
				// if lhs is identifier (new/call expression), allow to parse an assignment operator next
				// otherwise keep eating unary expressions and then any "value"
				// after that search for a binary operator. if we only ate a new/call expression then
				// also allow to eat assignments. repeat for the rhs.
				var parsedUnaryOperator = false;
				var isUnary = null;
				while (
					!isBreakOrContinueArg && // no unary for break/continue
					(isUnary =
						(match.value && this.hashUnaryKeywordStart[match.value[0]] && this.regexUnaryKeywords.test(match.value)) || // (match.value == 'delete' || match.value == 'void' || match.value == 'typeof' || match.value == 'new') ||
						(match.name == 11/*PUNCTUATOR*/ && this.regexUnaryOperators.test(match.value))
					)
				) {
					if (isUnary) match.isUnaryOp = true;
					if (this.ast) { //#ifdef FULL_AST
						// find parenthesis
						if (match.value == 'new') ++news;
					} //#endif

					match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					// ensure that it is in fact a valid lhs-start
					if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) match = this.failsafe('ExpectedAnotherExpressionRhs', match);
					// not allowed to parse assignment
					parsedUnaryOperator = true;
				};

				// if we parsed any kind of unary operator, we cannot be parsing a labeled statement
				if (parsedUnaryOperator) mayParseLabeledStatementInstead = false;

				// so now we know match is a valid lhs-start and not a unary operator
				// it must be a string, number, regex, identifier 
				// or the start of an object literal ({), array literal ([) or group operator (().

				var acceptAssignment = false;

				// take care of the "open" cases first (group, array, object)
				if (match.value == '(') {
					if (this.ast) { //#ifdef FULL_AST
						var groupStack = stack;
						stack = [];
						stack.desc = 'grouped';
						stack.nextBlack = match.tokposb;
						groupStack.push(stack);

						var lhp = match;

						match.isGroupStart = true;
					} //#endif
					match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) match = this.failsafe('GroupingShouldStartWithExpression', match);
					// keep parsing expressions as long as they are followed by a comma
					match = this.eatExpressions(false, match, stack);

					if (match.value != ')') match = this.failsafe('UnclosedGroupingOperator', match);
					if (this.ast) { //#ifdef FULL_AST
						match.twin = lhp;
						lhp.twin = match;

						match.isGroupStop = true;

						if (stack[stack.length-1].desc == 'expressions') {
							// create ref to this expression group to the opening paren
							lhp.expressionArg = stack[stack.length-1];
						}
					} //#endif
					match = this.tokenizer.storeCurrentAndFetchNextToken(true, match, stack); // might be div

					if (this.ast) { //#ifdef FULL_AST
						stack = groupStack;
					} //#endif
					// you can assign to group results. and as long as the group does not contain a comma (and valid ref), it will work too :)
					acceptAssignment = true;
				// there's an extra rule for [ namely that, it must start with an expression but after that, expressions are optional
				} else if (match.value == '[') {
					if (this.ast) { //#ifdef FULL_AST
						stack.sub = 'array literal';
						stack.hasArrayLiteral = true;
						var lhsb = match;

						match.isArrayLiteralStart = true;

						if (!this.scope.arrays) this.scope.arrays = [];
						match.arrayId = this.scope.arrays.length;
						this.scope.arrays.push(match);

						match.targetScope = this.scope;
					} //#endif
					// keep parsing expressions as long as they are followed by a comma
					match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

					// arrays may start with "elided" commas
					while (match.value == ',') match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

					var foundAtLeastOneComma = true; // for entry in while
					while (foundAtLeastOneComma && match.value != ']') {
						foundAtLeastOneComma = false;

						if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value)) && match.name != 14/*error*/) match = this.failsafe('ArrayShouldStartWithExpression', match);
						match = this.eatExpressions(false, match, stack, true);

						while (match.value == ',') {
							foundAtLeastOneComma = true;
							match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
						}
					}
					if (match.value != ']') {
						match = this.failsafe('UnclosedPropertyBracket', match);
					}
					if (this.ast) { //#ifdef FULL_AST
						match.twin = lhsb;
						lhsb.twin = match;

						match.isArrayLiteralStop = true;
					} //#endif
					match = this.tokenizer.storeCurrentAndFetchNextToken(true, match, stack); // might be div
					while (match.value == '++' || match.value == '--') {
						// gobble and ignore?
						this.failignore('InvalidPostfixOperandArray', match, stack);
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					}
				// object literals need seperate handling...
				} else if (match.value == '{') {
					if (this.ast) { //#ifdef FULL_AST
						stack.sub = 'object literal';
						stack.hasObjectLiteral = true;

						match.isObjectLiteralStart = true;

						if (!this.scope.objects) this.scope.objects = [];
						match.objectId = this.scope.objects.length;
						this.scope.objects.push(match);

						var targetObject = match;
						match.targetScope = this.scope;
	
						var lhc = match;
					} //#endif

					match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					if (match.name == 12/*eof*/) {
						match = this.failsafe('ObjectLiteralExpectsColonAfterName', match);
					}
					// ObjectLiteral
					// PropertyNameAndValueList

					while (match.value != '}' && match.name != 14/*error*/) { // will stop if next token is } or throw if not and no comma is found
						// expecting a string, number, or identifier
						//if (match.name != 5/*STRING_SINGLE*/ && match.name != 6/*STRING_DOUBLE*/ && match.name != 3/*NUMERIC_HEX*/ && match.name != 4/*NUMERIC_DEC*/ && match.name != 2/*IDENTIFIER*/) {
						// TOFIX: more specific errors depending on type...
						if (!match.isNumber && !match.isString && match.name != 2/*IDENTIFIER*/) {
							match = this.failsafe('IllegalPropertyNameToken', match);
						}

						if (this.ast) { //#ifdef FULL_AST
							var objLitStack = stack;
							stack = [];
							stack.desc = 'objlit pair';
							stack.isObjectLiteralPair = true;
							stack.nextBlack = match.tokposb;
							objLitStack.push(stack);

							var propNameStack = stack;
							stack = [];
							stack.desc = 'objlit pair name';
							stack.nextBlack = match.tokposb;
							propNameStack.push(stack);

							propNameStack.sub = 'data';

							var propName = match;
							propName.isPropertyName = true;
						} //#endif

						var getset = match.value;
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
						if (this.ast) { //#ifdef FULL_AST
							stack = propNameStack;
						} //#endif
						
						// for get/set we parse a function-like definition. but only if it's immediately followed by an identifier (otherwise it'll just be the property 'get' or 'set')
						if (getset == 'get') {
							// "get" PropertyName "(" ")" "{" FunctionBody "}"
							if (match.value == ':') {
								if (this.ast) { //#ifdef FULL_AST
									propName.isPropertyOf = targetObject;
								} //#endif
								match = this.eatObjectLiteralColonAndBody(match, stack);
							} else {
								if (this.ast) { //#ifdef FULL_AST
									match.isPropertyOf = targetObject;
									propNameStack.sub = 'getter';
									propNameStack.isAccessor = true;
								} //#endif
								// if (match.name != 2/*IDENTIFIER*/ && match.name != 5/*STRING_SINGLE*/ && match.name != 6/*STRING_DOUBLE*/ && match.name != 3/*NUMERIC_HEX*/ && match.name != 4/*NUMERIC_DEC*/) {
								if (!match.isNumber && !match.isString && match.name != 2/*IDENTIFIER*/) match = this.failsafe('IllegalGetterSetterNameToken', match, true);
								match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
								if (match.value != '(') match = this.failsafe('GetterSetterNameFollowedByOpenParen', match);
								if (this.ast) { //#ifdef FULL_AST
									var lhp = match;
								} //#endif
								match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
								if (match.value != ')') match = this.failsafe('GetterHasNoArguments', match);
								if (this.ast) { //#ifdef FULL_AST
									match.twin = lhp;
									lhp.twin = match;
								} //#endif
								match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
								match = this.eatFunctionBody(match, stack);
							}
						} else if (getset == 'set') {
							// "set" PropertyName "(" PropertySetParameterList ")" "{" FunctionBody "}"
							if (match.value == ':') {
								if (this.ast) { //#ifdef FULL_AST
									propName.isPropertyOf = targetObject;
								} //#endif
								match = this.eatObjectLiteralColonAndBody(match, stack);
							} else {
								if (this.ast) { //#ifdef FULL_AST
									match.isPropertyOf = targetObject;
									propNameStack.sub = 'setter';
									propNameStack.isAccessor = true;
								} //#endif
								if (!match.isNumber && !match.isString && match.name != 2/*IDENTIFIER*/) match = this.failsafe('IllegalGetterSetterNameToken', match);
								match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
								if (match.value != '(') match = this.failsafe('GetterSetterNameFollowedByOpenParen', match);
								if (this.ast) { //#ifdef FULL_AST
									var lhp = match;
								} //#endif
								match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
								if (match.name != 2/*IDENTIFIER*/) {
									if (match.value == ')') match = this.failsafe('SettersMustHaveArgument', match);
									else match = this.failsafe('IllegalSetterArgumentNameToken', match);
								}
								match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
								if (match.value != ')') {
									if (match.value == ',') match = this.failsafe('SettersOnlyGetOneArgument', match);
									else match = this.failsafe('SetterHeaderShouldHaveClosingParen', match);
								}
								if (this.ast) { //#ifdef FULL_AST
									match.twin = lhp;
									lhp.twin = match;
								} //#endif
								match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
								match = this.eatFunctionBody(match, stack);
							}
						} else {
							// PropertyName ":" AssignmentExpression
							if (this.ast) { //#ifdef FULL_AST
								propName.isPropertyOf = targetObject;
							} //#endif
							match = this.eatObjectLiteralColonAndBody(match, stack);
						}

						if (this.ast) { //#ifdef FULL_AST
							stack = objLitStack;
						} //#endif

						// one trailing comma allowed
						if (match.value == ',') {
							match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
							if (match.value == ',') match = this.failsafe('IllegalDoubleCommaInObjectLiteral', match);
						} else if (match.value != '}') match = this.failsafe('UnclosedObjectLiteral', match);

						// either the next token is } and the loop breaks or
						// the next token is the start of the next PropertyAssignment...
					}
					// closing curly
					if (this.ast) { //#ifdef FULL_AST
						match.twin = lhc;
						lhc.twin = match;

						match.isObjectLiteralStop = true;
					} //#endif

					match = this.tokenizer.storeCurrentAndFetchNextToken(true, match, stack); // next may be div
					while (match.value == '++' || match.value == '--') {
						this.failignore('InvalidPostfixOperandObject', match, stack);
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					}
				} else if (match.value == 'function') { // function expression
					if (this.ast) { //#ifdef FULL_AST
						var oldstack = stack;
						stack = [];
						stack.desc = 'func expr';
						stack.isFunction = true;
						stack.nextBlack = match.tokposb;
						if (!this.scope.functions) this.scope.functions = [];
						match.functionId = this.scope.functions.length;
						this.scope.functions.push(match);
						oldstack.push(stack);
						var oldscope = this.scope;
						// add new scope
						match.scope = stack.scope = this.scope = [
							this.scope,
							{value:'this', isDeclared:true, isEcma:true, functionStack: stack},
							{value:'arguments', isDeclared:true, isEcma:true, varType:['Object']}
						]; // add the current scope (to build chain up-down)
						this.scope.upper = oldscope;
						// ref to back to function that's the cause for this scope
						this.scope.scopeFor = match;
						match.targetScope = oldscope; // consistency
						match.isFuncExprKeyword = true;
						match.functionStack = stack;
					} //#endif
					var funcExprToken = match;

					match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					if (mayParseLabeledStatementInstead && match.value == ':') match = this.failsafe('LabelsMayNotBeReserved', match);
					if (match.name == 2/*IDENTIFIER*/) {
						funcExprToken.funcName = match;
						match.meta = "func expr name";
						match.varType = ['Function'];
						match.functionStack = stack; // ref to the stack, in case we detect the var being a constructor
						if (this.ast) { //#ifdef FULL_AST
							// name is only available to inner scope
							this.scope.push({value:match.value});
						} //#endif
						if (this.hashStartKeyOrReserved[match.value[0]] /*this.regexStartKeyOrReserved.test(match.value[0])*/ && this.regexIsKeywordOrReserved.test(match.value)) match = this.failsafe('FunctionNameMustNotBeReserved', match);
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					}
					match = this.eatFunctionParametersAndBody(match, stack, true, funcExprToken); // first token after func-expr is div

					while (match.value == '++' || match.value == '--') {
						this.failignore('InvalidPostfixOperandFunction', match, stack);
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					}

					if (this.ast) { //#ifdef FULL_AST
						// restore stack and scope
						stack = oldstack;
						this.scope = oldscope;
					} //#endif
				} else if (match.name <= 6) { // IDENTIFIER STRING_SINGLE STRING_DOUBLE NUMERIC_HEX NUMERIC_DEC REG_EX
					// save it in case it turns out to be a label.
					var possibleLabel = match;

					// validate the identifier, if any
					if (match.name == 2/*IDENTIFIER*/) {
						if (
							// this, null, true, false are actually allowed here
							!this.regexLiteralKeywords.test(match.value) &&
							// other reserved words are not
							this.hashStartKeyOrReserved[match.value[0]] /*this.regexStartKeyOrReserved.test(match.value[0])*/ && this.regexIsKeywordOrReserved.test(match.value)
						) {
							// if break/continue, we skipped the unary operator check so throw the proper error here
							if (isBreakOrContinueArg) {
								this.failignore('BreakOrContinueArgMustBeJustIdentifier', match, stack);
							} else if (match.value == 'else') {
								this.failignore('DidNotExpectElseHere', match, stack);
							} else {
								//if (mayParseLabeledStatementInstead) {new ZeParser.Error('LabelsMayNotBeReserved', match);
								// TOFIX: lookahead to see if colon is following. throw label error instead if that's the case
								// any forbidden keyword at this point is likely to be a statement start.
								// its likely that the parser will take a while to recover from this point...
								this.failignore('UnexpectedToken', match, stack);
								// TOFIX: maybe i should just return at this point. cut my losses and hope for the best.
							}
						}

						// only accept assignments after a member expression (identifier or ending with a [] suffix)
						acceptAssignment = true;
					} else if (isBreakOrContinueArg) match = this.failsafe('BreakOrContinueArgMustBeJustIdentifier', match);

					// the current match is the lead value being queried. tag it that way
					if (this.ast) { //#ifdef FULL_AST
						// dont mark labels
						if (!isBreakOrContinueArg) {
							match.meta = 'lead value';
							match.leadValue = true;
						}
					} //#endif


					// ok. gobble it.
					match = this.tokenizer.storeCurrentAndFetchNextToken(true, match, stack); // division allowed

					// now check for labeled statement (if mayParseLabeledStatementInstead then the first token for this expression must be an (unreserved) identifier)
					if (mayParseLabeledStatementInstead && match.value == ':') {
						if (possibleLabel.name != 2/*IDENTIFIER*/) {
							// label was not an identifier
							// TOFIX: this colon might be a different type of error... more analysis required
							this.failignore('LabelsMayOnlyBeIdentifiers', match, stack);
						}

						mayParseLabeledStatementInstead = true; // mark label parsed (TOFIX:speed?)
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

						possibleLabel.isLabel = true;
						if (this.ast) { //#ifdef FULL_AST
							delete possibleLabel.meta; // oh oops, it's not a lead value.

							possibleLabel.isLabelDeclaration = true;
							this.statementLabels.push(possibleLabel.value);

							stack.desc = 'labeled statement';
						} //#endif

						var errorIdToReplace = this.errorStack.length;
						// eat another statement now, its the body of the labeled statement (like if and while)
						match = this.eatStatement(false, match, stack);

						// if no statement was found, check here now and correct error
						if (match.error && match.error.msg == ZeParser.Errors.UnableToParseStatement.msg) {
							// replace with better error...
							match.error = new ZeParser.Error('LabelRequiresStatement');
							// also replace on stack
							this.errorStack[errorIdToReplace] = match.error;
						}

						match.wasLabel = true;

						return match;
					}

					mayParseLabeledStatementInstead = false;
				} else if (match.value == '}') {
					// ignore... its certainly the end of this expression, but maybe asi can be applied...
					// it might also be an object literal expecting more, but that case has been covered else where.
					// if it turns out the } is bad after all, .parse() will try to recover
				} else if (match.name == 14/*error*/) {
					do {
						if (match.tokenError) {
							var pe = new ZeParser.Error('TokenizerError', match);
							pe.msg += ': '+match.error.msg;
							this.errorStack.push(pe);
							
							this.failSpecial({start:match.start,stop:match.start,name:14/*error*/,error:pe}, match, stack)
						}
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
					} while (match.name == 14/*error*/);
				} else if (match.name == 12/*eof*/) {
					// cant parse any further. you're probably just typing...
					return match;
				} else {
					//if (!this.errorStack.length && match.name != 12/*eof*/) console.log(["unknown token", match, stack, Gui.escape(this.input)]);
					this.failignore('UnknownToken', match, stack);
					// we cant really ignore this. eat the token and try again. possibly you're just typing?
					match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
				}

				// search for "value" suffix. property access and call parens.
				while (match.value == '.' || match.value == '[' || match.value == '(') {
					if (isBreakOrContinueArg) match = this.failsafe('BreakOrContinueArgMustBeJustIdentifier', match);

					if (match.value == '.') {
						// property access. read in an IdentifierName (no keyword checks). allow assignments
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
						if (match.name != 2/*IDENTIFIER*/) this.failignore('PropertyNamesMayOnlyBeIdentifiers', match, stack);
						if (this.ast) { //#ifdef FULL_AST
							match.isPropertyName = true;
						} //#endif
						match = this.tokenizer.storeCurrentAndFetchNextToken(true, match, stack); // may parse div
						acceptAssignment = true;
					} else if (match.value == '[') {
						if (this.ast) { //#ifdef FULL_AST
							var lhsb = match;
							match.propertyAccessStart = true;
						} //#endif
						// property access, read expression list. allow assignments
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
						if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) {
							if (match.value == ']') match = this.failsafe('SquareBracketsMayNotBeEmpty', match);
							else match = this.failsafe('SquareBracketExpectsExpression', match);
						}
						match = this.eatExpressions(false, match, stack);
						if (match.value != ']') match = this.failsafe('UnclosedSquareBrackets', match);
						if (this.ast) { //#ifdef FULL_AST
							match.twin = lhsb;
							match.propertyAccessStop = true;
							lhsb.twin = match;

							if (stack[stack.length-1].desc == 'expressions') {
								// create ref to this expression group to the opening bracket
								lhsb.expressionArg = stack[stack.length-1];
							}
						} //#endif
						match = this.tokenizer.storeCurrentAndFetchNextToken(true, match, stack); // might be div
						acceptAssignment = true;
					} else if (match.value == '(') {
						if (this.ast) { //#ifdef FULL_AST
							var lhp = match;
							match.isCallExpressionStart = true;
							if (news) {
								match.parensBelongToNew = true;
								--news;
							}
						} //#endif
						// call expression, eat optional expression list, disallow assignments
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
						if (/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value)) match = this.eatExpressions(false, match, stack); // arguments are optional
						if (match.value != ')') match = this.failsafe('UnclosedCallParens', match);
						if (this.ast) { //#ifdef FULL_AST
							match.twin = lhp;
							lhp.twin = match;
							match.isCallExpressionStop = true;

							if (stack[stack.length-1].desc == 'expressions') {
								// create ref to this expression group to the opening bracket
								lhp.expressionArg = stack[stack.length-1];
							}
						} //#endif
						match = this.tokenizer.storeCurrentAndFetchNextToken(true, match, stack); // might be div
						acceptAssignment = false;
					}
				}

				// check for postfix operators ++ and --
				// they are stronger than the + or - binary operators
				// they can be applied to any lhs (even when it wouldnt make sense)
				// if there was a newline, it should get an ASI
				if ((match.value == '++' || match.value == '--') && !match.newline) {
					if (isBreakOrContinueArg) match = this.failsafe('BreakOrContinueArgMustBeJustIdentifier', match);
					match = this.tokenizer.storeCurrentAndFetchNextToken(true, match, stack); // may parse div
				}

				if (this.ast) { //#ifdef FULL_AST
					// restore "expression" stack
					stack = estack;
				} //#endif
				// now see if there is an operator following...

				do { // this do allows us to parse multiple ternary expressions in succession without screwing up.
					var ternary = false;
					if (
						(!forHeader && match.value == 'in') || // one of two named binary operators, may not be first expression in for-header (when semi's occur in the for-header)
						(match.value == 'instanceof') || // only other named binary operator
						((match.name == 11/*PUNCTUATOR*/) && // we can only expect a punctuator now
							(match.isAssignment = this.regexAssignments.test(match.value)) || // assignments are only okay with proper lhs
							this.regexNonAssignmentBinaryExpressionOperators.test(match.value) // test all other binary operators
						)
					) {
						if (match.isAssignment) {
							if (!acceptAssignment) this.failignore('IllegalLhsForAssignment', match, stack);
							else if (parsedNonAssignmentOperator) this.failignore('AssignmentNotAllowedAfterNonAssignmentInExpression', match, stack);
						}
						if (isBreakOrContinueArg) match = this.failsafe('BreakOrContinueArgMustBeJustIdentifier', match);

						if (!match.isAssignment) parsedNonAssignmentOperator = true; // last allowed assignment
						if (this.ast) { //#ifdef FULL_AST
							match.isBinaryOperator = true;
							// we build a stack to ensure any whitespace doesnt break the 1+(n*2) children rule for expressions
							var ostack = stack;
							stack = [];
							stack.desc = 'operator-expression';
							stack.isBinaryOperator = true;
							stack.sub = match.value;
							stack.nextBlack = match.tokposb;
							ostack.sub = match.value;
							stack.isAssignment = match.isAssignment;
							ostack.push(stack);
						} //#endif
						ternary = match.value == '?';
						// math, logic, assignment or in or instanceof
						match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

						if (this.ast) { //#ifdef FULL_AST
							// restore "expression" stack
							stack = ostack;
						} //#endif

						// minor exception to ternary operator, we need to parse two expressions nao. leave the trailing expression to the loop.
						if (ternary) {
							// LogicalORExpression "?" AssignmentExpression ":" AssignmentExpression
							// so that means just one expression center and right.
							if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) this.failignore('InvalidCenterTernaryExpression', match, stack);
							match = this.eatExpressions(false, match, stack, true, forHeader); // only one expression allowed inside ternary center/right

							if (match.value != ':') {
								if (match.value == ',') match = this.failsafe('TernarySecondExpressionCanNotContainComma', match);
								else match = this.failsafe('UnfinishedTernaryOperator', match);
							}
							if (this.ast) { //#ifdef FULL_AST
								// we build a stack to ensure any whitespace doesnt break the 1+(n*2) children rule for expressions
								var ostack = stack;
								stack = [];
								stack.desc = 'operator-expression';
								stack.sub = match.value;
								stack.nextBlack = match.tokposb;
								ostack.sub = match.value;
								stack.isAssignment = match.isAssignment;
								ostack.push(stack);
							} //#endif
							match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
							if (this.ast) { //#ifdef FULL_AST
								stack = ostack;
							} //#endif
							// rhs of the ternary can not contain a comma either
							match = this.eatExpressions(false, match, stack, true, forHeader); // only one expression allowed inside ternary center/right
						}
					} else {
						parseAnotherExpression = false;
					}
				} while (ternary); // if we just parsed a ternary expression, we need to check _again_ whether the next token is a binary operator.

				// start over. match is the rhs for the lhs we just parsed, but lhs for the next expression
				if (parseAnotherExpression && !(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) {
					// no idea what to do now. lets just ignore and see where it ends. TOFIX: maybe just break the loop or return?
					this.failignore('InvalidRhsExpression', match, stack);
				}
			}

			if (this.ast) { //#ifdef FULL_AST
				// restore "expressions" stack
				stack = astack;
			} //#endif

			// at this point we should have parsed one AssignmentExpression
			// lets see if we can parse another one...
			mayParseLabeledStatementInstead = first = false;
		} while (!onlyOne && match.value == ',');

		if (this.ast) { //#ifdef FULL_AST
			// remove empty array
			if (!stack.length) pstack.length = pstack.length-1;
			pstack.numberOfExpressions = parsedExpressions;
			if (pstack[0]) pstack[0].numberOfExpressions = parsedExpressions;
			stack.expressionCount = parsedExpressions;
		} //#endif
		return match;
	},
	eatFunctionDeclaration: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			var prevscope = this.scope;
			stack.desc = 'func decl';
			stack.isFunction = true;
			stack.nextBlack = match.tokposb;
			if (!this.scope.functions) this.scope.functions = [];
			match.functionId = this.scope.functions.length;
			this.scope.functions.push(match);
			// add new scope
			match.scope = stack.scope = this.scope = [
				this.scope, // add current scope (build scope chain up-down)
				// Object.create(null,
				{value:'this', isDeclared:true, isEcma:true, functionStack:stack},
				// Object.create(null,
				{value:'arguments', isDeclared:true, isEcma:true, varType:['Object']}
			];
			// ref to back to function that's the cause for this scope
			this.scope.scopeFor = match;
			match.targetScope = prevscope; // consistency
			
			match.functionStack = stack;

			match.isFuncDeclKeyword = true;
		} //#endif
		// only place that this function is used already checks whether next token is function
		var functionKeyword = match;
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.name != 2/*IDENTIFIER*/) match = this.failsafe('FunctionDeclarationsMustHaveName', match);
		if (this.hashStartKeyOrReserved[match.value[0]] /*this.regexStartKeyOrReserved.test(match.value[0])*/ && this.regexIsKeywordOrReserved.test(match.value)) this.failignore('FunctionNameMayNotBeReserved', match, stack);
		if (this.ast) { //#ifdef FULL_AST
			functionKeyword.funcName = match;
			prevscope.push({value:match.value});
			match.meta = 'func decl name'; // that's what it is, really
			match.varType = ['Function'];
			match.functionStack = stack;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatFunctionParametersAndBody(match, stack, false, functionKeyword); // first token after func-decl is regex
		if (this.ast) { //#ifdef FULL_AST
			// restore previous scope
			this.scope = prevscope;
		} //#endif
		return match;
	},
	eatObjectLiteralColonAndBody: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			var propValueStack = stack;
			stack = [];
			stack.desc = 'objlit pair colon';
			stack.nextBlack = match.tokposb;
			propValueStack.push(stack);
		} //#endif
		if (match.value != ':') match = this.failsafe('ObjectLiteralExpectsColonAfterName', match);
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (this.ast) { //#ifdef FULL_AST
			stack = propValueStack;
		} //#endif

		// this might actually fail due to ASI optimization.
		// if the property name does not exist and it is the last item
		// of the objlit, the expression parser will see an unexpected
		// } and ignore it, giving some leeway to apply ASI. of course,
		// that doesnt work for objlits. but we dont want to break the
		// existing mechanisms. so we check this differently... :)
		var prevMatch = match;
		match = this.eatExpressions(false, match, stack, true); // only one expression
		if (match == prevMatch) match = this.failsafe('ObjectLiteralMissingPropertyValue', match);

		return match;
	},
	eatFunctionParametersAndBody: function(match, stack, div, funcToken){
		// div: the first token _after_ a function expression may be a division...
		if (match.value != '(') match = this.failsafe('ExpectingFunctionHeaderStart', match);
		else if (this.ast) { //#ifdef FULL_AST
			var lhp = match;
			funcToken.lhp = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.name == 2/*IDENTIFIER*/) { // params
			if (this.hashStartKeyOrReserved[match.value[0]] /*this.regexStartKeyOrReserved.test(match.value[0])*/ && this.regexIsKeywordOrReserved.test(match.value)) this.failignore('FunctionArgumentsCanNotBeReserved', match, stack);
			if (this.ast) { //#ifdef FULL_AST
				if (!funcToken.paramNames) funcToken.paramNames = [];
				stack.paramNames = funcToken.paramNames;
				funcToken.paramNames.push(match);
				this.scope.push({value:match.value}); // add param name to scope
				match.meta = 'parameter';
			} //#endif
			match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
			while (match.value == ',') {
				match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
				if (match.name != 2/*IDENTIFIER*/) {
					// example: if name is 12, the source is incomplete...
					this.failignore('FunctionParametersMustBeIdentifiers', match, stack);
				} else if (this.hashStartKeyOrReserved[match.value[0]] /*this.regexStartKeyOrReserved.test(match.value[0])*/ && this.regexIsKeywordOrReserved.test(match.value)) {
					this.failignore('FunctionArgumentsCanNotBeReserved', match, stack);
				}
				if (this.ast) { //#ifdef FULL_AST
					// Object.create(null,
					this.scope.push({value:match.value}); // add param name to scope
					match.meta = 'parameter';
					if (match.name == 2/*IDENTIFIER*/) funcToken.paramNames.push(match);
				} //#endif
				match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
			}
		}
		if (this.ast) { //#ifdef FULL_AST
			if (lhp) {
				match.twin = lhp;
				lhp.twin = match;
				funcToken.rhp = match;
			}
		} //#endif
		if (match.value != ')') match = this.failsafe('ExpectedFunctionHeaderClose', match); // TOFIX: can be various things here...
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatFunctionBody(match, stack, div, funcToken);
		return match;
	},
	eatFunctionBody: function(match, stack, div, funcToken){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'func body';
			stack.nextBlack = match.tokposb;

			// create EMPTY list of functions. labels cannot cross function boundaries
			var labelBackup = this.statementLabels;
			this.statementLabels = [];
			stack.labels = this.statementLabels;
		} //#endif

		// if div, a division can occur _after_ this function expression
		//this.stats.eatFunctionBody = (+//this.stats.eatFunctionBody||0)+1;
		if (match.value != '{') match = this.failsafe('ExpectedFunctionBodyCurlyOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhc = match;
			if (funcToken) funcToken.lhc = lhc;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatSourceElements(match, stack);
		if (match.value != '}') match = this.failsafe('ExpectedFunctionBodyCurlyClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhc;
			lhc.twin = match;
			if (funcToken) funcToken.rhc = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(div, match, stack);

		if (this.ast) { //#ifdef FULL_AST
			// restore label set
			this.statementLabels = labelBackup;
		} //#endif

		return match;
	},
	eatVar: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'var';
			stack.nextBlack = match.tokposb;
			match.stack = stack;
			match.isVarKeyword = true;
		} //#endif
		match = this.eatVarDecl(match, stack);
		match = this.eatSemiColon(match, stack);

		return match;
	},
	eatVarDecl: function(match, stack, forHeader){
		// assumes match is indeed the identifier 'var'
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'var decl';
			stack.nextBlack = match.tokposb;

			var targetScope = this.scope;
			while (targetScope.catchScope) targetScope = targetScope[0];
		} //#endif
		var first = true;
		var varsDeclared = 0;
		do {
			++varsDeclared;
			match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack); // start: var, iteration: comma
			if (this.ast) { //#ifdef FULL_AST
				var declStack = stack;
				var stack = [];
				stack.desc = 'single var decl';
				stack.varStack = declStack; // reference to the var statement stack, it might hook to jsdoc needed for these vars
				stack.nextBlack = match.tokposb;
				declStack.push(stack);

				var singleDecStack = stack;
				stack = [];
				stack.desc = 'sub-expression';
				stack.nextBlack = match.tokposb;
				singleDecStack.push(stack);
			} //#endif

			// next token should be a valid identifier
			if (match.name == 12/*eof*/) {
				if (first) match = this.failsafe('VarKeywordMissingName', match);
				// else, ignore. TOFIX: return?
				else match = this.failsafe('IllegalTrailingComma', match);
			} else if (match.name != 2/*IDENTIFIER*/) {
				match = this.failsafe('VarNamesMayOnlyBeIdentifiers', match);
			} else if (this.hashStartKeyOrReserved[match.value[0]] /*this.regexStartKeyOrReserved.test(match.value[0])*/ && this.regexIsKeywordOrReserved.test(match.value)) {
				match = this.failsafe('VarNamesCanNotBeReserved', match);
			}
			// mark the match as being a variable name. we need it for lookup later :)
			if (this.ast) { //#ifdef FULL_AST
				match.meta = 'var name';
				targetScope.push({value:match.value});
			} //#endif
			match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

			if (this.ast) { //#ifdef FULL_AST
				stack = singleDecStack;
			} //#endif

			// next token should either be a = , or ;
			// if = parse an expression and optionally a comma
			if (match.value == '=') {
				if (this.ast) { //#ifdef FULL_AST
					singleDecStack = stack;
					stack = [];
					stack.desc = 'operator-expression';
					stack.sub = '=';
					stack.nextBlack = match.tokposb;
					singleDecStack.push(stack);

					stack.isAssignment = true;
				} //#endif
				match.isInitialiser = true;
				match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
				if (this.ast) { //#ifdef FULL_AST
					stack = singleDecStack;
				} //#endif

				if (!(/*is left hand side start?*/ match.name <= 6 || match.name == 14/*error*/ || this.regexLhsStart.test(match.value))) match = this.failsafe('VarInitialiserExpressionExpected', match);
				match = this.eatExpressions(false, match, stack, true, forHeader); // only one expression 
				// var statement: comma or semi now
				// for statement: semi, comma or 'in'
			}
			if (this.ast) { //#ifdef FULL_AST
				stack = declStack;
			} //#endif

			// determines proper error message in one case
			first = false;
		// keep parsing name(=expression) sequences as long as you see a comma here
		} while (match.value == ',');

		if (this.ast) { //#ifdef FULL_AST
			stack.varsDeclared = varsDeclared;
		} //#endif

		return match;
	},

	eatIf: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'if';
			stack.hasElse = false;
			stack.nextBlack = match.tokposb;
		} //#endif
		// (
		// expression
		// )
		// statement
		// [else statement]
		var ifKeyword = match;
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '(') match = this.failsafe('ExpectedStatementHeaderOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhp = match;
			match.statementHeaderStart = true;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) match = this.failsafe('StatementHeaderIsNotOptional', match);
		match = this.eatExpressions(false, match, stack);
		if (match.value != ')') match = this.failsafe('ExpectedStatementHeaderClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhp;
			match.statementHeaderStop = true;
			lhp.twin = match;

			if (stack[stack.length-1].desc == 'expressions') {
				// create ref to this expression group to the opening bracket
				lhp.expressionArg = stack[stack.length-1];
			}
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatStatement(false, match, stack);

		// match might be null here... (if the if-statement was end part of the source)
		if (match && match.value == 'else') {
			if (this.ast) { //#ifdef FULL_AST
				ifKeyword.hasElse = match;
			} //#endif
			match = this.eatElse(match, stack);
		}

		return match;
	},
	eatElse: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.hasElse = true;
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'else';
			stack.nextBlack = match.tokposb;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatStatement(false, match, stack);

		return match;
	},
	eatDo: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'do';
			stack.isIteration = true;
			stack.nextBlack = match.tokposb;
			this.statementLabels.push(''); // add "empty"
			var doToken = match;
		} //#endif
		// statement
		// while
		// (
		// expression
		// )
		// semi-colon
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatStatement(false, match, stack);
		if (match.value != 'while') match = this.failsafe('DoShouldBeFollowedByWhile', match);
		if (this.ast) { //#ifdef FULL_AST
			match.hasDo = doToken;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '(') match = this.failsafe('ExpectedStatementHeaderOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhp = match;
			match.statementHeaderStart = true;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) match = this.failsafe('StatementHeaderIsNotOptional', match);
		match = this.eatExpressions(false, match, stack);
		if (match.value != ')') match = this.failsafe('ExpectedStatementHeaderClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhp;
			match.statementHeaderStop = true;
			match.isForDoWhile = true; // prevents missing block warnings
			lhp.twin = match;

			if (stack[stack.length-1].desc == 'expressions') {
				// create ref to this expression group to the opening bracket
				lhp.expressionArg = stack[stack.length-1];
			}
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatSemiColon(match, stack); // TOFIX: this is not optional according to the spec, but browsers apply ASI anyways

		return match;
	},
	eatWhile: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'while';
			stack.isIteration = true;
			stack.nextBlack = match.tokposb;
			this.statementLabels.push(''); // add "empty"
		} //#endif

		// (
		// expression
		// )
		// statement
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '(') match = this.failsafe('ExpectedStatementHeaderOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhp = match;
			match.statementHeaderStart = true;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) match = this.failsafe('StatementHeaderIsNotOptional', match);
		match = this.eatExpressions(false, match, stack);
		if (match.value != ')') match = this.failsafe('ExpectedStatementHeaderClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhp;
			match.statementHeaderStop = true;
			lhp.twin = match;

			if (stack[stack.length-1].desc == 'expressions') {
				// create ref to this expression group to the opening bracket
				lhp.expressionArg = stack[stack.length-1];
			}
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatStatement(false, match, stack);

		return match;
	},

	eatFor: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'for';
			stack.isIteration = true;
			stack.nextBlack = match.tokposb;
			this.statementLabels.push(''); // add "empty"
		} //#endif
		// either a for(..in..) or for(..;..;..)
		// start eating an expression but refuse to parse
		// 'in' on the top-level of that expression. they are fine
		// in sub-levels (group, array, etc). Now the expression
		// must be followed by either ';' or 'in'. Else throw.
		// Branch on that case, ; requires two.
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '(') match = this.failsafe('ExpectedStatementHeaderOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhp = match;
			match.statementHeaderStart = true;
			match.forHeaderStart = true;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		// for (either case) may start with var, in which case you'll parse a var declaration before encountering the 'in' or first semi.
		if (match.value == 'var') {
			match = this.eatVarDecl(match, stack, true);
		} else if (match.value != ';') { // expressions are optional in for-each
			if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) {
				this.failignore('StatementHeaderIsNotOptional', match, stack);
			}
			match = this.eatExpressions(false, match, stack, false, true); // can parse multiple expressions, in is not ok here
		}

		// now we parsed an expression if it existed. the next token should be either ';' or 'in'. branch accordingly
		if (match.value == 'in') {
			var declStack = stack[stack.length-1];
			if (declStack.varsDeclared > 1) {
				// disallowed. for-in var decls can only have one var name declared
				this.failignore('ForInCanOnlyDeclareOnVar', match, stack);
			}
			
			if (this.ast) { //#ifdef FULL_AST
				stack.forType = 'in';
				match.forFor = true; // make easy distinction between conditional and iterational operator
			} //#endif

			// just parse another expression, where 'in' is allowed.
			match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
			match = this.eatExpressions(false, match, stack);
		} else {
			if (match.value != ';') match = this.failsafe('ForHeaderShouldHaveSemisOrIn', match);

			if (this.ast) { //#ifdef FULL_AST
				stack.forType = 'each';
				match.forEachHeaderStart = true;
			} //#endif
			// parse another optional no-in expression, another semi and then one more optional no-in expression
			match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
			if (/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value)) match = this.eatExpressions(false, match, stack); // in is ok here
			if (match.value != ';') match = this.failsafe('ExpectedSecondSemiOfForHeader', match);
			if (this.ast) { //#ifdef FULL_AST
				match.forEachHeaderStop = true;
			} //#endif
			match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
			if (/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value)) match = this.eatExpressions(false, match, stack); // in is ok here
		}

		if (match.value != ')') match = this.failsafe('ExpectedStatementHeaderClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhp;
			match.statementHeaderStop = true;
			match.forHeaderStop = true;
			lhp.twin = match;

			if (match.forType == 'in' && stack[stack.length-1].desc == 'expressions') {
				// create ref to this expression group to the opening bracket
				lhp.expressionArg = stack[stack.length-1];
			}
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		match = this.eatStatement(false, match, stack);

		return match;
	},
	eatContinue: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'continue';
			stack.nextBlack = match.tokposb;

			match.restricted = true;
		} //#endif
		// (no-line-break identifier)
		// ;
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack); // may not have line terminator...
		if (!match.newline && match.value != ';' && match.name != 12/*EOF*/ && match.value != '}') {
			if (this.ast) { //#ifdef FULL_AST
				match.isLabel = true;
				match.isLabelTarget = true;

				var continueArg = match; // remember to see if this continue parsed a label
			} //#endif
			// may only parse exactly an identifier at this point
			match = this.eatExpressions(false, match, stack, true, false, true); // first true=onlyOne, second: continue/break arg
			if (this.ast) { //#ifdef FULL_AST
				stack.hasLabel = continueArg != match;
			} //#endif
			if (match.value != ';' && !match.newline && match.name != 12/*eof*/ && match.value != '}') match = this.failsafe('BreakOrContinueArgMustBeJustIdentifier', match);
		}
		match = this.eatSemiColon(match, stack);

		return match;
	},
	eatBreak: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			var parentstack = stack
			stack = [];
			stack.desc = 'statement';
			stack.sub = 'break';
			stack.nextBlack = match.tokposb;
			
			parentstack.push(stack);

			match.restricted = true;
		} //#endif
		// (no-line-break identifier)
		// ;
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack); // may not have line terminator...
		if (!match.newline && match.value != ';' && match.name != 12/*EOF*/ && match.value != '}') {
			if (this.ast) { //#ifdef FULL_AST
				match.isLabel = true;
				match.isLabelTarget = true;
				var breakArg = match; // remember to see if this break parsed a label
			} //#endif
			// may only parse exactly an identifier at this point
			match = this.eatExpressions(false, match, stack, true, false, true); // first true=onlyOne, second: continue/break arg
			if (this.ast) { //#ifdef FULL_AST
				stack.hasLabel = breakArg != match;
			} //#endif

			if (match.value != ';' && !match.newline && match.name != 12/*eof*/ && match.value != '}') match = this.failsafe('BreakOrContinueArgMustBeJustIdentifier', match);
		}
		match = this.eatSemiColon(match, stack);

		return match;
	},
	eatReturn: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'return';
			stack.nextBlack = match.tokposb;
			stack.returnFor = this.scope.scopeFor;

			match.restricted = true;
		} //#endif
		// (no-line-break expression)
		// ;
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack); // may not have line terminator...
		if (!match.newline && match.value != ';' && match.name != 12/*EOF*/ && match.value != '}') {
			match = this.eatExpressions(false, match, stack);
		}
		match = this.eatSemiColon(match, stack);

		return match;
	},
	eatThrow: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'throw';
			stack.nextBlack = match.tokposb;

			match.restricted = true;
		} //#endif
		// (no-line-break expression)
		// ;
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack); // may not have line terminator...
		if (match.newline) match = this.failsafe('ThrowCannotHaveReturn', match);
		if (match.value == ';') match = this.failsafe('ThrowMustHaveArgument', match);
		match = this.eatExpressions(false, match, stack);
		match = this.eatSemiColon(match, stack);

		return match;
	},
	eatSwitch: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'switch';
			stack.nextBlack = match.tokposb;

			this.statementLabels.push(''); // add "empty"
		} //#endif
		// meh.
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '(') match = this.failsafe('ExpectedStatementHeaderOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhp = match;
			match.statementHeaderStart = true;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) {
			this.failignore('StatementHeaderIsNotOptional', match, stack);
		}
		match = this.eatExpressions(false, match, stack);
		if (match.value != ')') match = this.failsafe('ExpectedStatementHeaderClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhp;
			match.statementHeaderStop = true;
			lhp.twin = match;

			if (stack[stack.length-1].desc == 'expressions') {
				// create ref to this expression group to the opening bracket
				lhp.expressionArg = stack[stack.length-1];
			}
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '{') match = this.failsafe('SwitchBodyStartsWithCurly', match);

		if (this.ast) { //#ifdef FULL_AST
			var lhc = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		// you may parse a default case, and only once per switch. but you may do so anywhere.
		var parsedAnything = false;

		while (match.value == 'case' || (!stack.parsedSwitchDefault && match.value == 'default')) {
			parsedAnything = true;

			match = this.eatSwitchClause(match, stack);
		}

		// if you didnt parse anything but not encountering a closing curly now, you might be thinking that switches may start with silly stuff
		if (!parsedAnything && match.value != '}') {
			match = this.failsafe('SwitchBodyMustStartWithClause', match);
		}

		if (stack.parsedSwitchDefault && match.value == 'default') {
			this.failignore('SwitchCannotHaveDoubleDefault', match, stack);
		}

		if (match.value != '}' && match.name != 14/*error*/) match = this.failsafe('SwitchBodyEndsWithCurly', match);

		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhc;
			lhc.twin = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		return match;
	},
	eatSwitchClause: function(match, stack){
		match = this.eatSwitchHeader(match, stack);
		match = this.eatSwitchBody(match, stack);

		return match;
	},
	eatSwitchHeader: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			// collect whitespace...
			var switchHeaderStack = stack
			stack.push(stack = []);
			stack.desc = 'switch clause header';
			stack.nextBlack = match.tokposb;
		} //#endif

		if (match.value == 'case') {
			match = this.eatSwitchCaseHead(match, stack);
		} else { // default
			if (this.ast) { //#ifdef FULL_AST
				switchHeaderStack.hasDefaultClause = true;
			} //#endif
			match = this.eatSwitchDefaultHead(match, stack);
		}

		if (this.ast) { //#ifdef FULL_AST
			// just to group whitespace (makes certain navigation easier..)
			stack.push(stack = []);
			stack.desc = 'colon';
			stack.nextBlack = match.tokposb;
		} //#endif

		if (match.value != ':') {
			match = this.failsafe('SwitchClausesEndWithColon', match);
		}
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		return match;
	},
	eatSwitchBody: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'switch clause body';
			stack.nextBlack = match.tokposb;
		} //#endif

		// parse body of case or default, just so long case and default keywords are not seen and end of switch is not reached
		// (clause bodies may be empty, for instance to fall through)
		var lastMatch = null;
		while (match.value != 'default' && match.value != 'case' && match.value != '}' && match.name != 14/*error*/ && match.name != 12/*eof*/ && lastMatch != match) {
			lastMatch = match; // prevents endless loops on error ;)
			match = this.eatStatement(true, match, stack);
		}
		if (lastMatch == match) this.failsafe('UnexpectedInputSwitch', match);

		return match;
	},
	eatSwitchCaseHead: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.sub = 'case';
			var caseHeadStack = stack;

			stack.push(stack = []);
			stack.desc = 'case';
			stack.nextBlack = match.tokposb;

			match.isCase = true;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		if (match.value == ':') {
			this.failignore('CaseMissingExpression', match, stack);
		} else {
			if (this.ast) { //#ifdef FULL_AST
				caseHeadStack.push(stack = []);
				stack.desc = 'case arg';
				stack.nextBlack = match.tokposb;
			} //#endif
			match = this.eatExpressions(false, match, stack);
		}

		return match;
	},
	eatSwitchDefaultHead: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.sub = 'default';

			stack.push(stack = []);
			stack.desc = 'case';
			stack.nextBlack = match.tokposb;

			match.isDefault = true;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		return match;
	},
	eatTryCatchFinally: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'try';
			stack.nextBlack = match.tokposb;
		} //#endif

		match = this.eatTry(match, stack);

		if (match.value == 'catch') {
			if (this.ast) { //#ifdef FULL_AST
				stack.hasCatch = true;
			} //#endif
			match = this.eatCatch(match, stack);
		}
		if (match.value == 'finally') {
			if (this.ast) { //#ifdef FULL_AST
				stack.hasFinally = true;
			} //#endif
			match = this.eatFinally(match, stack);
		}

		// at least a catch or finally block must follow. may be both.
		if (!stack.tryHasCatchOrFinally) {
			this.failignore('TryMustHaveCatchOrFinally', match, stack);
		}

		return match;
	},
	eatTry: function(match, stack){
		// block
		// (catch ( identifier ) block )
		// (finally block)
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '{') match = this.failsafe('MissingTryBlockCurlyOpen', match);

		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'tryblock';
			stack.nextBlack = match.tokposb;
			var lhc = match;
		} //#endif

		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '}') match = this.eatStatements(match, stack);
		if (match.value != '}') match = this.failsafe('MissingTryBlockCurlyClose', match);

		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhc;
			lhc.twin = match;
		} //#endif
		
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		return match;
	},
	eatCatch: function(match, stack){
		stack.tryHasCatchOrFinally = true;
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'catch';
			stack.nextBlack = match.tokposb;

			// the catch block has a header which can contain at most one parameter
			// this parameter is bound to a local stack. formally, if that parameter
			// shadows another variable, changes made to the variable inside the catch
			// should not be reflected by the variable being shadowed. however, this
			// is not very safe to rely on so there ought to be a warning. note that
			// only this parameter gets bound to this inner scope, other parameters.

			var catchScopeBackup = this.scope;
			match.scope = this.scope = stack.scope = [this.scope];
			this.scope.catchScope = true; // mark this as being a catchScope

			// find first function scope or global scope object...
			var nonCatchScope = catchScopeBackup;
			while (nonCatchScope.catchScope) nonCatchScope = nonCatchScope[0];

			// get catch id, which is governed by the function/global scope only
			if (!nonCatchScope.catches) nonCatchScope.catches = [];
			match.catchId = nonCatchScope.catches.length;
			nonCatchScope.catches.push(match);
			match.targetScope = nonCatchScope;
			match.catchScope = this.scope;

			// ref to back to function that's the cause for this scope
			this.scope.scopeFor = match;
			// catch clauses dont have a special `this` or `arguments`, map them to their parent scope
			if (catchScopeBackup.global) this.scope.push(catchScopeBackup[0]); // global (has no `arguments` but always a `this`)
			else if (catchScopeBackup.catchScope) {
				// tricky. there will at least be a this
				this.scope.push(catchScopeBackup[1]);
				// but there might not be an arguments
				if (catchScopeBackup[2] && catchScopeBackup[2].value == 'arguments') this.scope.push(catchScopeBackup[2]);
			} else this.scope.push(catchScopeBackup[1], catchScopeBackup[2]); // function scope, copy this and arguments
		} //#endif

		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '(') match = this.failsafe('CatchHeaderMissingOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhp = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.name != 2/*IDENTIFIER*/) match = this.failsafe('MissingCatchParameter', match);
		if (this.hashStartKeyOrReserved[match.value[0]] /*this.regexStartKeyOrReserved.test(match.value[0])*/ && this.regexIsKeywordOrReserved.test(match.value)) {
			this.failignore('CatchParameterNameMayNotBeReserved', match, stack);
		}

		if (this.ast) { //#ifdef FULL_AST
			match.meta = 'var name';
			// this is the catch variable. bind it to a scope but keep the scope as
			// it currently is.
			this.scope.push(match);
			match.isCatchVar = true;
		} //#endif

		// now the catch body will use the outer scope to bind new variables. the problem is that
		// inner scopes, if any, should have access to the scope variable, so their scope should
		// be linked to the catch scope. this is a problem in the current architecture but the 
		// idea is to pass on the catchScope as the scope to the eatStatements call, etc.

		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != ')') match = this.failsafe('CatchHeaderMissingClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhp;
			lhp.twin = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '{') match = this.failsafe('MissingCatchBlockCurlyOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhc = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		// catch body. statements are optional.	
		if (match.value != '}') match = this.eatStatements(match, stack);

		if (match.value != '}') match = this.failsafe('MissingCatchBlockCurlyClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhc;
			lhc.twin = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		if (this.ast) { //#ifdef FULL_AST
			this.scope = catchScopeBackup;
		} //#endif

		return match;
	},
	eatFinally: function(match, stack){
		stack.tryHasCatchOrFinally = true;
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'finally';
			stack.nextBlack = match.tokposb;
		} //#endif

		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '{') match = this.failsafe('MissingFinallyBlockCurlyOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhc = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '}') match = this.eatStatements(match, stack);
		if (match.value != '}') match = this.failsafe('MissingFinallyBlockCurlyClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhc;
			lhc.twin = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		return match;
	},
	eatDebugger: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'debugger';
			stack.nextBlack = match.tokposb;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatSemiColon(match, stack);

		return match;
	},
	eatWith: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.push(stack = []);
			stack.desc = 'statement';
			stack.sub = 'with';
			stack.nextBlack = match.tokposb;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (match.value != '(') match = this.failsafe('ExpectedStatementHeaderOpen', match);
		if (this.ast) { //#ifdef FULL_AST
			var lhp = match;
			match.statementHeaderStart = true;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		if (!(/*is left hand side start?*/ match.name <= 6 || this.regexLhsStart.test(match.value))) match = this.failsafe('StatementHeaderIsNotOptional', match);
		match = this.eatExpressions(false, match, stack);
		if (match.value != ')') match = this.failsafe('ExpectedStatementHeaderClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhp;
			match.statementHeaderStop = true;
			lhp.twin = match;

			if (stack[stack.length-1].desc == 'expressions') {
				// create ref to this expression group to the opening bracket
				lhp.expressionArg = stack[stack.length-1];
			}
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);
		match = this.eatStatement(false, match, stack);

		return match;
	},
	eatFunction: function(match, stack){
		var pe = new ZeParser.Error
		this.errorStack.push(pe);
		// ignore. browsers will accept it anyways
		var error = {start:match.stop,stop:match.stop,name:14/*error*/,error:pe};
		this.specialError(error, match, stack);
		// now try parsing a function declaration...
		match = this.eatFunctionDeclaration(match, stack);

		return match;
	},
	eatLabelOrExpression: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			var parentstack = stack;

			stack = [];
			stack.desc = 'statement';
			stack.sub = 'expression';
			stack.nextBlack = match.tokposb;
			parentstack.push(stack);
		} //#endif
		// must be an expression or a labeled statement.
		// in order to prevent very weird return constructs, we'll first check the first match
		// if that's an identifier, we'll gobble it here and move on to the second.
		// if that's a colon, we'll gobble it as a labeled statement. otherwise, we'll pass on
		// control to eatExpression, with the note that we've already gobbled a 

		match = this.eatExpressions(true, match, stack);
		// if we parsed a label, the returned match (colon) will have this property
		if (match.wasLabel) {
			if (this.ast) { //#ifdef FULL_AST
				stack.sub = 'labeled';
			} //#endif
			// it will have already eaten another statement for the label
		} else {
			if (this.ast) { //#ifdef FULL_AST
				stack.sub = 'expression';
			} //#endif
			// only parse semi if we didnt parse a label just now...
			match = this.eatSemiColon(match, stack);
		}

		return match;
	},
	eatBlock: function(match, stack){
		if (this.ast) { //#ifdef FULL_AST
			stack.sub = 'block';
			var lhc = match;
		} //#endif

		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		if (match.value == '}') {
			if (this.ast) { //#ifdef FULL_AST
				stack.isEmptyBlock = true;
			} //#endif
		} else {
			match = this.eatStatements(match, stack);
		}
		if (match.value != '}') match = this.failsafe('BlockCurlyClose', match);
		if (this.ast) { //#ifdef FULL_AST
			match.twin = lhc;
			lhc.twin = match;
		} //#endif
		match = this.tokenizer.storeCurrentAndFetchNextToken(false, match, stack);

		return match;
	},

	eatStatements: function(match, stack){
		//this.stats.eatStatements = (+//this.stats.eatStatements||0)+1;
		// detecting the start of a statement "quickly" is virtually impossible.
		// instead we keep eating statements until the match stops changing
		// the first argument indicates that the statement is optional. if that
		// statement was not found, the input match will also be the output.

		while (match != (match = this.eatStatement(true, match, stack)));
		return match;
	},
	eatStatement: function(isOptional, match, stack){
		if (!match && isOptional) return match; // eof

		if (this.ast) { //#ifdef FULL_AST
			match.statementStart = true;
			var pstack = stack;
			stack = [];
			stack.desc = 'statement-parent';
			stack.nextBlack = match.tokposb;
			pstack.push(stack);

			// list of labels, these are bound to statements (and can access any label higher up, but not cross functions)
			var labelBackup = this.statementLabels;
			this.statementLabels = [labelBackup]; // make ref like tree. we need this to catch labels parsed beyond the current position (not yet known to use)
			stack.labels = this.statementLabels;
		} //#endif

		if (match.name == 2/*IDENTIFIER*/) {
			// try to determine whether it's a statement
			// (block/empty statements come later, this branch is only for identifiers)
			switch (match.value) {
				case 'var':
					match = this.eatVar(match, stack);
					break;
				case 'if':
					match = this.eatIf(match, stack);
					break;
				case 'do':
					match = this.eatDo(match, stack);
					break;
				case 'while':
					match = this.eatWhile(match, stack);
					break;
				case 'for':
					match = this.eatFor(match, stack);
					break;
				case 'continue':
					match = this.eatContinue(match, stack);
					break;
				case 'break':
					match = this.eatBreak(match, stack);
					break;
				case 'return':
					match = this.eatReturn(match, stack);
					break;
				case 'throw':
					match = this.eatThrow(match, stack);
					break;
				case 'switch':
					match = this.eatSwitch(match, stack);
					break;
				case 'try':
					match = this.eatTryCatchFinally(match, stack);
					break;
				case 'debugger':
					match = this.eatDebugger(match, stack);
					break;
				case 'with':
					match = this.eatWith(match, stack);
					break;
				case 'function':
					// I'm not sure whether this is at all possible.... (but it's bad, either way ;)
					// so add an error token, but parse the function as if it was a declaration.
					this.failignore('StatementMayNotStartWithFunction', match, stack);

					// now parse as declaration... (most likely?)
					match = this.eatFunctionDeclaration(match, stack);

					break;
				default: // either a label or an expression-statement
					match = this.eatLabelOrExpression(match, stack);
			}
		} else if (match.value == '{') { // Block (make sure you do this before checking for expression...)
			match = this.eatBlock(match, stack);
		} else if (
			// expression statements:
			match.isString ||
			match.isNumber ||
			match.name == 1/*REG_EX*/ ||
			this.regexLhsStart.test(match.value)
		) {
			match = this.eatExpressions(false, match,stack);
			match = this.eatSemiColon(match, stack);
		} else if (match.value == ';') { // empty statement
			match.emptyStatement = true;
			match = this.eatSemiColon(match, stack);
		} else if (!isOptional) {
			if (this.ast) { //#ifdef FULL_AST
				// unmark token as being start of a statement, since it's obviously not
				match.statementStart = false;
			} //#endif
			match = this.failsafe('UnableToParseStatement', match);
		} else {
			// unmark token as being start of a statement, since it's obviously not
			if (this.ast) match.statementStart = true;
		}

		if (this.ast) { //#ifdef FULL_AST
			if (!stack.length) pstack.length = pstack.length-1;

			// restore label set
			this.statementLabels = labelBackup;
		} //#endif

		return match;
	},

	eatSourceElements: function(match, stack){
		//this.stats.eatSourceElements = (+//this.stats.eatSourceElements||0)+1;
		// detecting the start of a statement "quickly" is virtually impossible.
		// instead we keep eating statements until the match stops changing
		// the first argument indicates that the statement is optional. if that
		// statement was not found, the input match will also be the output.
		while (match != oldMatch) { // difficult to determine whether ` && match.name != 12/*EOF*/` is actually speeding things up. it's an extra check vs one less call to eatStatement...
			var oldMatch = match;
			// always try to eat function declaration first. otherwise 'function' at the start might cause eatStatement to throw up
			if (match.value == 'function') match = this.eatFunctionDeclaration(match, stack);
			else match = this.eatStatement(true, match, stack);
		}
		return match;
	},

	failsafe: function(name, match, doNotAddMatch){
		var pe = new ZeParser.Error(name, match);
		this.errorStack.push(pe);

		if (!doNotAddMatch) {
			// the match was bad, but add it to the ast anyways. in most cases this is the case but in some its not.
			// the tokenizer will pick up on the errorEscape property and add it after the match we passed on.
			if (this.tokenizer.errorEscape) this.stack.push(this.tokenizer.errorEscape);
			this.tokenizer.errorEscape = match;
		}
		var error = {start:match.start,stop:match.start,len:0, name:14/*error*/,error:pe, value:''};
		this.tokenizer.addTokenToStreamBefore(error, match);
		return error;
	},
	failignore: function(name, match, stack){
		var pe = new ZeParser.Error(name, match);
		this.errorStack.push(pe);
		// ignore the error (this will screw up :)
		var error = {start:match.start,stop:match.start,len:0,name:14/*error*/,error:pe, value:''};
		stack.push(error);
		this.tokenizer.addTokenToStreamBefore(error, match);
	},
	failSpecial: function(error, match, stack){
		// we cant really ignore this. eat the token
		stack.push(error);
		this.tokenizer.addTokenToStreamBefore(error, match);
	},

0:0};

//#ifdef TEST_SUITE
ZeParser.testSuite = function(tests){
	var ok = 0;
	var fail = 0;
	var start = +new Date;
	for (var i = 0; i < tests.length; ++i) {
		var test = tests[i], input = test[0], desc = test[test.length - 1], stack = [];
		try {
			new ZeParser(input, new Tokenizer(input), stack).parse();
			++ok;
		} catch (e) {
			++fail;
		}
		document.getElementsByTagName('div')[0].innerHTML = ('Ze parser test suite finished ('+(+new Date - start)+' ms). ok:'+ok+', fail:'+fail);
	};
};
//#endif

ZeParser.regexLhsStart = /[\+\-\~\!\(\{\[]/;
/*
ZeParser.regexStartKeyword = /[bcdefinrstvw]/;
ZeParser.regexKeyword = /^break$|^catch$|^continue$|^debugger$|^default$|^delete$|^do$|^else$|^finally$|^for$|^function$|^if$|^in$|^instanceof$|^new$|^return$|^switch$|^this$|^throw$|^try$|^typeof$|^var$|^void$|^while$|^with$/;
ZeParser.regexStartReserved = /[ceis]/;
ZeParser.regexReserved = /^class$|^const$|^enum$|^export$|^extends$|^import$|^super$/;
*/
ZeParser.regexStartKeyOrReserved = /[bcdefinrstvw]/;
ZeParser.hashStartKeyOrReserved = Object.create ? Object.create(null, {b:{value:1},c:{value:1},d:{value:1},e:{value:1},f:{value:1},i:{value:1},n:{value:1},r:{value:1},s:{value:1},t:{value:1},v:{value:1},w:{value:1}}) : {b:1,c:1,d:1,e:1,f:1,i:1,n:1,r:1,s:1,t:1,v:1,w:1};
ZeParser.regexIsKeywordOrReserved = /^break$|^catch$|^continue$|^debugger$|^default$|^delete$|^do$|^else$|^finally$|^for$|^function$|^if$|^in$|^instanceof$|^new$|^return$|^switch$|^case$|^this$|^true$|^false$|^null$|^throw$|^try$|^typeof$|^var$|^void$|^while$|^with$|^class$|^const$|^enum$|^export$|^extends$|^import$|^super$/;
ZeParser.regexAssignments = /^[\+\-\*\%\&\|\^\/]?=$|^\<\<\=$|^\>{2,3}\=$/;
ZeParser.regexNonAssignmentBinaryExpressionOperators = /^[\+\-\*\%\|\^\&\?\/]$|^[\<\>]\=?$|^[\=\!]\=\=?$|^\<\<|\>\>\>?$|^\&\&$|^\|\|$/;
ZeParser.regexUnaryKeywords = /^delete$|^void$|^typeof$|^new$/;
ZeParser.hashUnaryKeywordStart = Object.create ? Object.create(null, {d:{value:1},v:{value:1},t:{value:1},n:{value:1}}) : {d:1,v:1,t:1,n:1};
ZeParser.regexUnaryOperators = /[\+\-\~\!]/;
ZeParser.regexLiteralKeywords = /^this$|^null$|^true$|^false$/;

ZeParser.Error = function(type, match){
	//if (type == 'BreakOrContinueArgMustBeJustIdentifier') throw here;
	this.msg = ZeParser.Errors[type].msg;
	this.before = ZeParser.Errors[type].before;
	this.match = match;
};

ZeParser.Errors = {
	NoASI: {msg:'Expected semi-colon, was unable to apply ASI'},
	ExpectedAnotherExpressionComma: {msg:'expecting another (left hand sided) expression after the comma'},
	ExpectedAnotherExpressionRhs: {msg:"expected a rhs expression"},
	UnclosedGroupingOperator: {msg:"Unclosed grouping operator"},
	GroupingShouldStartWithExpression: {msg:'The grouping operator (`(`) should start with a left hand sided expression'},
	ArrayShouldStartWithExpression: {msg:'The array literal (`[`) should start with a left hand sided expression'},
	UnclosedPropertyBracket: {msg:'Property bracket was not closed after expression (expecting `]`)'},
	IllegalPropertyNameToken: {msg:'Object literal property names can only be assigned as strings, numbers or identifiers'},
	IllegalGetterSetterNameToken: {msg:'Name of a getter/setter can only be assigned as strings, numbers or identifiers'},
	GetterSetterNameFollowedByOpenParen: {msg:'The name of the getter/setter should immediately be followed by the opening parenthesis `(`'},
	GetterHasNoArguments: {msg:'The opening parenthesis `(` of the getter should be immediately followed by the closing parenthesis `)`, the getter cannot have an argument'},
	IllegalSetterArgumentNameToken: {msg:'Expecting the name of the argument of a setter, can only be assigned as strings, numbers or identifiers'},
	SettersOnlyGetOneArgument: {msg:'Setters have one and only one argument, missing the closing parenthesis `)`'},
	SetterHeaderShouldHaveClosingParen: {msg:'After the first argument of a setter should come a closing parenthesis `)`'},
	SettersMustHaveArgument: {msg:'Setters must have exactly one argument defined'},
	UnclosedObjectLiteral: {msg:'Expected to find a comma `,` for the next expression or a closing curly brace `}` to end the object literal'},
	FunctionNameMustNotBeReserved: {msg:'Function name may not be a keyword or a reserved word'},
	ExpressionMayNotStartWithKeyword: {msg:'Expressions may not start with keywords or reserved words that are not in this list: [this, null, true, false, void, typeof, delete, new]'},
	LabelsMayOnlyBeIdentifiers: {msg:'Label names may only be defined as an identifier'},
	LabelsMayNotBeReserved: {msg:'Labels may not be a keyword or a reserved word'},
	UnknownToken: {msg:'Unknown token encountered, dont know how to proceed'},
	PropertyNamesMayOnlyBeIdentifiers: {msg:'The tokens of property names accessed through the dot operator may only be identifiers'},
	SquareBracketExpectsExpression: {msg:'The square bracket property access expects an expression'},
	SquareBracketsMayNotBeEmpty: {msg:'Square brackets may never be empty, expecting an expression'},
	UnclosedSquareBrackets: {msg:'Unclosed square bracket encountered, was expecting `]` after the expression'},
	UnclosedCallParens: {msg:'Unclosed call parenthesis, expecting `)` after the optional expression'},
	InvalidCenterTernaryExpression: {msg:'Center expression of ternary operator should be a regular expression (but may not contain the comma operator directly)'},
	UnfinishedTernaryOperator: {msg:'Encountered a ternary operator start (`?`) but did not find the required colon (`:`) after the center expression'},
	TernarySecondExpressionCanNotContainComma: {msg:'The second and third expressions of the ternary operator can/may not "directly" contain a comma operator'},
	InvalidRhsExpression: {msg:'Expected a right hand side expression after the operator (which should also be a valid lhs) but did not find one'},
	FunctionDeclarationsMustHaveName: {msg:'Function declaration must have name'},
	FunctionNameMayNotBeReserved: {msg:'Function name may not be a keyword or reserved word'},
	ExpectingFunctionHeaderStart: {msg:'Expected the opening parenthesis of the function header'},
	FunctionArgumentsCanNotBeReserved: {msg:'Function arguments may not be keywords or reserved words'},
	FunctionParametersMustBeIdentifiers: {msg:'Function arguments must be identifiers'},
	ExpectedFunctionHeaderClose: {msg:'Expected the closing parenthesis `)` of the function header'},
	ExpectedFunctionBodyCurlyOpen: {msg:'Expected the opening curly brace `{` for the function body'},
	ExpectedFunctionBodyCurlyClose: {msg:'Expected the closing curly brace `}` for the function body'},
	VarNamesMayOnlyBeIdentifiers: {msg:'Missing variable name, must be a proper identifier'},
	VarNamesCanNotBeReserved: {msg:'Variable names may not be keywords or reserved words'},
	VarInitialiserExpressionExpected: {msg:'The initialiser of the variable statement should be an expression without comma'},
	ExpectedStatementHeaderOpen: {msg:'Expected opening parenthesis `(` for statement header'},
	StatementHeaderIsNotOptional: {msg:'Statement header must not be empty'},
	ExpectedStatementHeaderClose: {msg:'Expected closing parenthesis `)` for statement header'},
	DoShouldBeFollowedByWhile: {msg:'The do-while statement requires the `while` keyword after the expression'},
	ExpectedSecondSemiOfForHeader: {msg:'Expected the second semi-colon of the for-each header'},
	ForHeaderShouldHaveSemisOrIn: {msg:'The for-header should contain at least the `in` operator or two semi-colons (`;`)'},
	SwitchBodyStartsWithCurly: {msg:'The body of a switch statement starts with a curly brace `{`'},
	SwitchClausesEndWithColon: {msg:'Switch clauses (`case` and `default`) end with a colon (`:`)'},
	SwitchCannotHaveDoubleDefault: {msg:'Switches cannot have more than one `default` clause'},
	SwitchBodyEndsWithCurly: {msg:'The body of a switch statement ends with a curly brace `}`'},
	MissingTryBlockCurlyOpen: {msg:'Missing the opening curly brace (`{`) for the block of the try statement'},
	MissingTryBlockCurlyClose: {msg:'Missing the closing curly brace (`}`) for the block of the try statement'},
	CatchHeaderMissingOpen: {msg:'Missing the opening parenthesis of the catch header'},
	MissingCatchParameter: {msg:'Catch clauses should have exactly one argument which will be bound to the error object being thrown'},
	CatchParameterNameMayNotBeReserved: {msg:'Catch clause parameter may not be a keyword or reserved word'},
	CatchHeaderMissingClose: {msg:'Missing the closing parenthesis of the catch header'},
	MissingCatchBlockCurlyOpen: {msg:'Missing the opening curly brace (`{`) for the block of the catch statement'},
	MissingCatchBlockCurlyClose: {msg:'Missing the closing curly brace (`}`) for the block of the catch statement'},
	MissingFinallyBlockCurlyOpen: {msg:'Missing the opening curly brace (`{`) for the block of the finally statement'},
	MissingFinallyBlockCurlyClose: {msg:'Missing the closing curly brace (`}`) for the block of the finally statement'},
	StatementMayNotStartWithFunction: {msg:'statements may not start with function...', before:true},
	BlockCurlyClose: {msg:'Expected the closing curly (`}`) for a block statement'},
	BlockCurlyOpen: {msg:'Expected the closing curly (`}`) for a block statement'},
	UnableToParseStatement: {msg:'Was unable to find a statement when it was requested'},
	IllegalDoubleCommaInObjectLiteral: {msg:'A double comma in object literals is not allowed'},
	ObjectLiteralExpectsColonAfterName: {msg:'After every property name (identifier, string or number) a colon (`:`) should follow'},
	ThrowMustHaveArgument: {msg:'The expression argument for throw is not optional'},
	ThrowCannotHaveReturn: {msg:'There may not be a return between throw and the start of its expression argument'},
	SwitchBodyMustStartWithClause: {msg:'The body of a switch clause must start with at a case or default clause (but may be empty, which would be silly)'},
	BreakOrContinueArgMustBeJustIdentifier: {msg:'The argument to a break or continue statement must be exactly and only an identifier (an existing label)'},
	AssignmentNotAllowedAfterNonAssignmentInExpression: {msg:'An assignment is not allowed if it is preceeded by a non-expression operator in the same expression-level'},
	IllegalLhsForAssignment: {msg:'Illegal left hand side for assignment (you cannot assign to things like string literals, number literals or function calls}'},
	VarKeywordMissingName: {msg:'Var keyword should be followed by a variable name'},
	IllegalTrailingComma: {msg:'Illegal trailing comma found'},
	ObjectLiteralMissingPropertyValue: {msg:'Missing object literal property value'},
	TokenizerError: {msg:'Tokenizer encountered unexpected input'},
	LabelRequiresStatement: {msg:'Saw a label without the (required) statement following'},
	DidNotExpectElseHere: {msg:'Did not expect an else here. To what if should it belong? Maybe you put a ; after the if-block? (if(x){};else{})'},
	UnexpectedToken: {msg:'Found an unexpected token and have no idea why'},
	InvalidPostfixOperandArray: {msg:'You cannot apply ++ or -- to an array'},
	InvalidPostfixOperandObject: {msg:'You cannot apply ++ or -- to an object'},
	InvalidPostfixOperandFunction: {msg:'You cannot apply ++ or -- to a function'},
	CaseMissingExpression: {msg:'Case expects an expression before the colon'},
	TryMustHaveCatchOrFinally: {msg:'The try statement must have a catch or finally block'},
	UnexpectedInputSwitch: {msg:'Unexpected input while parsing a switch clause...'},
	ForInCanOnlyDeclareOnVar: {msg:'For-in header can only introduce one new variable'}
};
