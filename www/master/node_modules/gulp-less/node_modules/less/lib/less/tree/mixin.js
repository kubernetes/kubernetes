(function (tree) {

tree.mixin = {};
tree.mixin.Call = function (elements, args, index, currentFileInfo, important) {
    this.selector = new(tree.Selector)(elements);
    this.arguments = (args && args.length) ? args : null;
    this.index = index;
    this.currentFileInfo = currentFileInfo;
    this.important = important;
};
tree.mixin.Call.prototype = {
    type: "MixinCall",
    accept: function (visitor) {
        if (this.selector) {
            this.selector = visitor.visit(this.selector);
        }
        if (this.arguments) {
            this.arguments = visitor.visitArray(this.arguments);
        }
    },
    eval: function (env) {
        var mixins, mixin, args, rules = [], match = false, i, m, f, isRecursive, isOneFound, rule,
            candidates = [], candidate, conditionResult = [], defaultFunc = tree.defaultFunc,
            defaultResult, defNone = 0, defTrue = 1, defFalse = 2, count, originalRuleset; 

        args = this.arguments && this.arguments.map(function (a) {
            return { name: a.name, value: a.value.eval(env) };
        });

        for (i = 0; i < env.frames.length; i++) {
            if ((mixins = env.frames[i].find(this.selector)).length > 0) {
                isOneFound = true;
                
                // To make `default()` function independent of definition order we have two "subpasses" here.
                // At first we evaluate each guard *twice* (with `default() == true` and `default() == false`),
                // and build candidate list with corresponding flags. Then, when we know all possible matches,
                // we make a final decision.
                
                for (m = 0; m < mixins.length; m++) {
                    mixin = mixins[m];
                    isRecursive = false;
                    for(f = 0; f < env.frames.length; f++) {
                        if ((!(mixin instanceof tree.mixin.Definition)) && mixin === (env.frames[f].originalRuleset || env.frames[f])) {
                            isRecursive = true;
                            break;
                        }
                    }
                    if (isRecursive) {
                        continue;
                    }
                    
                    if (mixin.matchArgs(args, env)) {  
                        candidate = {mixin: mixin, group: defNone};
                        
                        if (mixin.matchCondition) { 
                            for (f = 0; f < 2; f++) {
                                defaultFunc.value(f);
                                conditionResult[f] = mixin.matchCondition(args, env);
                            }
                            if (conditionResult[0] || conditionResult[1]) {
                                if (conditionResult[0] != conditionResult[1]) {
                                    candidate.group = conditionResult[1] ?
                                        defTrue : defFalse;
                                }

                                candidates.push(candidate);
                            }   
                        }
                        else {
                            candidates.push(candidate);
                        }
                        
                        match = true;
                    }
                }
                
                defaultFunc.reset();

                count = [0, 0, 0];
                for (m = 0; m < candidates.length; m++) {
                    count[candidates[m].group]++;
                }

                if (count[defNone] > 0) {
                    defaultResult = defFalse;
                } else {
                    defaultResult = defTrue;
                    if ((count[defTrue] + count[defFalse]) > 1) {
                        throw { type: 'Runtime',
                            message: 'Ambiguous use of `default()` found when matching for `'
                                + this.format(args) + '`',
                            index: this.index, filename: this.currentFileInfo.filename };
                    }
                }
                
                for (m = 0; m < candidates.length; m++) {
                    candidate = candidates[m].group;
                    if ((candidate === defNone) || (candidate === defaultResult)) {
                        try {
                            mixin = candidates[m].mixin;
                            if (!(mixin instanceof tree.mixin.Definition)) {
                                originalRuleset = mixin.originalRuleset || mixin;
                                mixin = new tree.mixin.Definition("", [], mixin.rules, null, false);
                                mixin.originalRuleset = originalRuleset;
                            }
                            Array.prototype.push.apply(
                                  rules, mixin.evalCall(env, args, this.important).rules);
                        } catch (e) {
                            throw { message: e.message, index: this.index, filename: this.currentFileInfo.filename, stack: e.stack };
                        }
                    }
                }
                
                if (match) {
                    if (!this.currentFileInfo || !this.currentFileInfo.reference) {
                        for (i = 0; i < rules.length; i++) {
                            rule = rules[i];
                            if (rule.markReferenced) {
                                rule.markReferenced();
                            }
                        }
                    }
                    return rules;
                }
            }
        }
        if (isOneFound) {
            throw { type:    'Runtime',
                    message: 'No matching definition was found for `' + this.format(args) + '`',
                    index:   this.index, filename: this.currentFileInfo.filename };
        } else {
            throw { type:    'Name',
                    message: this.selector.toCSS().trim() + " is undefined",
                    index:   this.index, filename: this.currentFileInfo.filename };
        }
    },
    format: function (args) {
        return this.selector.toCSS().trim() + '(' +
            (args ? args.map(function (a) {
                var argValue = "";
                if (a.name) {
                    argValue += a.name + ":";
                }
                if (a.value.toCSS) {
                    argValue += a.value.toCSS();
                } else {
                    argValue += "???";
                }
                return argValue;
            }).join(', ') : "") + ")";
    }
};

tree.mixin.Definition = function (name, params, rules, condition, variadic, frames) {
    this.name = name;
    this.selectors = [new(tree.Selector)([new(tree.Element)(null, name, this.index, this.currentFileInfo)])];
    this.params = params;
    this.condition = condition;
    this.variadic = variadic;
    this.arity = params.length;
    this.rules = rules;
    this._lookups = {};
    this.required = params.reduce(function (count, p) {
        if (!p.name || (p.name && !p.value)) { return count + 1; }
        else                                 { return count; }
    }, 0);
    this.parent = tree.Ruleset.prototype;
    this.frames = frames;
};
tree.mixin.Definition.prototype = {
    type: "MixinDefinition",
    accept: function (visitor) {
        if (this.params && this.params.length) {
            this.params = visitor.visitArray(this.params);
        }
        this.rules = visitor.visitArray(this.rules);
        if (this.condition) {
            this.condition = visitor.visit(this.condition);
        }
    },
    variable:  function (name) { return this.parent.variable.call(this, name); },
    variables: function ()     { return this.parent.variables.call(this); },
    find:      function ()     { return this.parent.find.apply(this, arguments); },
    rulesets:  function ()     { return this.parent.rulesets.apply(this); },

    evalParams: function (env, mixinEnv, args, evaldArguments) {
        /*jshint boss:true */
        var frame = new(tree.Ruleset)(null, null),
            varargs, arg,
            params = this.params.slice(0),
            i, j, val, name, isNamedFound, argIndex, argsLength = 0;

        mixinEnv = new tree.evalEnv(mixinEnv, [frame].concat(mixinEnv.frames));

        if (args) {
            args = args.slice(0);
            argsLength = args.length;

            for(i = 0; i < argsLength; i++) {
                arg = args[i];
                if (name = (arg && arg.name)) {
                    isNamedFound = false;
                    for(j = 0; j < params.length; j++) {
                        if (!evaldArguments[j] && name === params[j].name) {
                            evaldArguments[j] = arg.value.eval(env);
                            frame.prependRule(new(tree.Rule)(name, arg.value.eval(env)));
                            isNamedFound = true;
                            break;
                        }
                    }
                    if (isNamedFound) {
                        args.splice(i, 1);
                        i--;
                        continue;
                    } else {
                        throw { type: 'Runtime', message: "Named argument for " + this.name +
                            ' ' + args[i].name + ' not found' };
                    }
                }
            }
        }
        argIndex = 0;
        for (i = 0; i < params.length; i++) {
            if (evaldArguments[i]) { continue; }

            arg = args && args[argIndex];

            if (name = params[i].name) {
                if (params[i].variadic) {
                    varargs = [];
                    for (j = argIndex; j < argsLength; j++) {
                        varargs.push(args[j].value.eval(env));
                    }
                    frame.prependRule(new(tree.Rule)(name, new(tree.Expression)(varargs).eval(env)));
                } else {
                    val = arg && arg.value;
                    if (val) {
                        val = val.eval(env);
                    } else if (params[i].value) {
                        val = params[i].value.eval(mixinEnv);
                        frame.resetCache();
                    } else {
                        throw { type: 'Runtime', message: "wrong number of arguments for " + this.name +
                            ' (' + argsLength + ' for ' + this.arity + ')' };
                    }
                    
                    frame.prependRule(new(tree.Rule)(name, val));
                    evaldArguments[i] = val;
                }
            }

            if (params[i].variadic && args) {
                for (j = argIndex; j < argsLength; j++) {
                    evaldArguments[j] = args[j].value.eval(env);
                }
            }
            argIndex++;
        }

        return frame;
    },
    eval: function (env) {
        return new tree.mixin.Definition(this.name, this.params, this.rules, this.condition, this.variadic, this.frames || env.frames.slice(0));
    },
    evalCall: function (env, args, important) {
        var _arguments = [],
            mixinFrames = this.frames ? this.frames.concat(env.frames) : env.frames,
            frame = this.evalParams(env, new(tree.evalEnv)(env, mixinFrames), args, _arguments),
            rules, ruleset;

        frame.prependRule(new(tree.Rule)('@arguments', new(tree.Expression)(_arguments).eval(env)));

        rules = this.rules.slice(0);

        ruleset = new(tree.Ruleset)(null, rules);
        ruleset.originalRuleset = this;
        ruleset = ruleset.eval(new(tree.evalEnv)(env, [this, frame].concat(mixinFrames)));
        if (important) {
            ruleset = this.parent.makeImportant.apply(ruleset);
        }
        return ruleset;
    },
    matchCondition: function (args, env) {
        if (this.condition && !this.condition.eval(
            new(tree.evalEnv)(env,
                [this.evalParams(env, new(tree.evalEnv)(env, this.frames ? this.frames.concat(env.frames) : env.frames), args, [])] // the parameter variables
                    .concat(this.frames) // the parent namespace/mixin frames
                    .concat(env.frames)))) { // the current environment frames
            return false;
        }
        return true;
    },
    matchArgs: function (args, env) {
        var argsLength = (args && args.length) || 0, len;

        if (! this.variadic) {
            if (argsLength < this.required)                               { return false; }
            if (argsLength > this.params.length)                          { return false; }
        } else {
            if (argsLength < (this.required - 1))                         { return false; }
        }

        len = Math.min(argsLength, this.arity);

        for (var i = 0; i < len; i++) {
            if (!this.params[i].name && !this.params[i].variadic) {
                if (args[i].value.eval(env).toCSS() != this.params[i].value.eval(env).toCSS()) {
                    return false;
                }
            }
        }
        return true;
    }
};

})(require('../tree'));
