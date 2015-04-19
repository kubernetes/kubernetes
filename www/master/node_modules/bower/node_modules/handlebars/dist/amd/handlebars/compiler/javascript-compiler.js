define(
  ["../base","../exception","exports"],
  function(__dependency1__, __dependency2__, __exports__) {
    "use strict";
    var COMPILER_REVISION = __dependency1__.COMPILER_REVISION;
    var REVISION_CHANGES = __dependency1__.REVISION_CHANGES;
    var Exception = __dependency2__["default"];

    function Literal(value) {
      this.value = value;
    }

    function JavaScriptCompiler() {}

    JavaScriptCompiler.prototype = {
      // PUBLIC API: You can override these methods in a subclass to provide
      // alternative compiled forms for name lookup and buffering semantics
      nameLookup: function(parent, name /* , type*/) {
        if (JavaScriptCompiler.isValidJavaScriptVariableName(name)) {
          return parent + "." + name;
        } else {
          return parent + "['" + name + "']";
        }
      },
      depthedLookup: function(name) {
        this.aliases.lookup = 'this.lookup';

        return 'lookup(depths, "' + name + '")';
      },

      compilerInfo: function() {
        var revision = COMPILER_REVISION,
            versions = REVISION_CHANGES[revision];
        return [revision, versions];
      },

      appendToBuffer: function(string) {
        if (this.environment.isSimple) {
          return "return " + string + ";";
        } else {
          return {
            appendToBuffer: true,
            content: string,
            toString: function() { return "buffer += " + string + ";"; }
          };
        }
      },

      initializeBuffer: function() {
        return this.quotedString("");
      },

      namespace: "Handlebars",
      // END PUBLIC API

      compile: function(environment, options, context, asObject) {
        this.environment = environment;
        this.options = options;
        this.stringParams = this.options.stringParams;
        this.trackIds = this.options.trackIds;
        this.precompile = !asObject;

        this.name = this.environment.name;
        this.isChild = !!context;
        this.context = context || {
          programs: [],
          environments: []
        };

        this.preamble();

        this.stackSlot = 0;
        this.stackVars = [];
        this.aliases = {};
        this.registers = { list: [] };
        this.hashes = [];
        this.compileStack = [];
        this.inlineStack = [];

        this.compileChildren(environment, options);

        this.useDepths = this.useDepths || environment.depths.list.length || this.options.compat;

        var opcodes = environment.opcodes,
            opcode,
            i,
            l;

        for (i = 0, l = opcodes.length; i < l; i++) {
          opcode = opcodes[i];

          this[opcode.opcode].apply(this, opcode.args);
        }

        // Flush any trailing content that might be pending.
        this.pushSource('');

        /* istanbul ignore next */
        if (this.stackSlot || this.inlineStack.length || this.compileStack.length) {
          throw new Exception('Compile completed with content left on stack');
        }

        var fn = this.createFunctionContext(asObject);
        if (!this.isChild) {
          var ret = {
            compiler: this.compilerInfo(),
            main: fn
          };
          var programs = this.context.programs;
          for (i = 0, l = programs.length; i < l; i++) {
            if (programs[i]) {
              ret[i] = programs[i];
            }
          }

          if (this.environment.usePartial) {
            ret.usePartial = true;
          }
          if (this.options.data) {
            ret.useData = true;
          }
          if (this.useDepths) {
            ret.useDepths = true;
          }
          if (this.options.compat) {
            ret.compat = true;
          }

          if (!asObject) {
            ret.compiler = JSON.stringify(ret.compiler);
            ret = this.objectLiteral(ret);
          }

          return ret;
        } else {
          return fn;
        }
      },

      preamble: function() {
        // track the last context pushed into place to allow skipping the
        // getContext opcode when it would be a noop
        this.lastContext = 0;
        this.source = [];
      },

      createFunctionContext: function(asObject) {
        var varDeclarations = '';

        var locals = this.stackVars.concat(this.registers.list);
        if(locals.length > 0) {
          varDeclarations += ", " + locals.join(", ");
        }

        // Generate minimizer alias mappings
        for (var alias in this.aliases) {
          if (this.aliases.hasOwnProperty(alias)) {
            varDeclarations += ', ' + alias + '=' + this.aliases[alias];
          }
        }

        var params = ["depth0", "helpers", "partials", "data"];

        if (this.useDepths) {
          params.push('depths');
        }

        // Perform a second pass over the output to merge content when possible
        var source = this.mergeSource(varDeclarations);

        if (asObject) {
          params.push(source);

          return Function.apply(this, params);
        } else {
          return 'function(' + params.join(',') + ') {\n  ' + source + '}';
        }
      },
      mergeSource: function(varDeclarations) {
        var source = '',
            buffer,
            appendOnly = !this.forceBuffer,
            appendFirst;

        for (var i = 0, len = this.source.length; i < len; i++) {
          var line = this.source[i];
          if (line.appendToBuffer) {
            if (buffer) {
              buffer = buffer + '\n    + ' + line.content;
            } else {
              buffer = line.content;
            }
          } else {
            if (buffer) {
              if (!source) {
                appendFirst = true;
                source = buffer + ';\n  ';
              } else {
                source += 'buffer += ' + buffer + ';\n  ';
              }
              buffer = undefined;
            }
            source += line + '\n  ';

            if (!this.environment.isSimple) {
              appendOnly = false;
            }
          }
        }

        if (appendOnly) {
          if (buffer || !source) {
            source += 'return ' + (buffer || '""') + ';\n';
          }
        } else {
          varDeclarations += ", buffer = " + (appendFirst ? '' : this.initializeBuffer());
          if (buffer) {
            source += 'return buffer + ' + buffer + ';\n';
          } else {
            source += 'return buffer;\n';
          }
        }

        if (varDeclarations) {
          source = 'var ' + varDeclarations.substring(2) + (appendFirst ? '' : ';\n  ') + source;
        }

        return source;
      },

      // [blockValue]
      //
      // On stack, before: hash, inverse, program, value
      // On stack, after: return value of blockHelperMissing
      //
      // The purpose of this opcode is to take a block of the form
      // `{{#this.foo}}...{{/this.foo}}`, resolve the value of `foo`, and
      // replace it on the stack with the result of properly
      // invoking blockHelperMissing.
      blockValue: function(name) {
        this.aliases.blockHelperMissing = 'helpers.blockHelperMissing';

        var params = [this.contextName(0)];
        this.setupParams(name, 0, params);

        var blockName = this.popStack();
        params.splice(1, 0, blockName);

        this.push('blockHelperMissing.call(' + params.join(', ') + ')');
      },

      // [ambiguousBlockValue]
      //
      // On stack, before: hash, inverse, program, value
      // Compiler value, before: lastHelper=value of last found helper, if any
      // On stack, after, if no lastHelper: same as [blockValue]
      // On stack, after, if lastHelper: value
      ambiguousBlockValue: function() {
        this.aliases.blockHelperMissing = 'helpers.blockHelperMissing';

        // We're being a bit cheeky and reusing the options value from the prior exec
        var params = [this.contextName(0)];
        this.setupParams('', 0, params, true);

        this.flushInline();

        var current = this.topStack();
        params.splice(1, 0, current);

        this.pushSource("if (!" + this.lastHelper + ") { " + current + " = blockHelperMissing.call(" + params.join(", ") + "); }");
      },

      // [appendContent]
      //
      // On stack, before: ...
      // On stack, after: ...
      //
      // Appends the string value of `content` to the current buffer
      appendContent: function(content) {
        if (this.pendingContent) {
          content = this.pendingContent + content;
        }

        this.pendingContent = content;
      },

      // [append]
      //
      // On stack, before: value, ...
      // On stack, after: ...
      //
      // Coerces `value` to a String and appends it to the current buffer.
      //
      // If `value` is truthy, or 0, it is coerced into a string and appended
      // Otherwise, the empty string is appended
      append: function() {
        // Force anything that is inlined onto the stack so we don't have duplication
        // when we examine local
        this.flushInline();
        var local = this.popStack();
        this.pushSource('if (' + local + ' != null) { ' + this.appendToBuffer(local) + ' }');
        if (this.environment.isSimple) {
          this.pushSource("else { " + this.appendToBuffer("''") + " }");
        }
      },

      // [appendEscaped]
      //
      // On stack, before: value, ...
      // On stack, after: ...
      //
      // Escape `value` and append it to the buffer
      appendEscaped: function() {
        this.aliases.escapeExpression = 'this.escapeExpression';

        this.pushSource(this.appendToBuffer("escapeExpression(" + this.popStack() + ")"));
      },

      // [getContext]
      //
      // On stack, before: ...
      // On stack, after: ...
      // Compiler value, after: lastContext=depth
      //
      // Set the value of the `lastContext` compiler value to the depth
      getContext: function(depth) {
        this.lastContext = depth;
      },

      // [pushContext]
      //
      // On stack, before: ...
      // On stack, after: currentContext, ...
      //
      // Pushes the value of the current context onto the stack.
      pushContext: function() {
        this.pushStackLiteral(this.contextName(this.lastContext));
      },

      // [lookupOnContext]
      //
      // On stack, before: ...
      // On stack, after: currentContext[name], ...
      //
      // Looks up the value of `name` on the current context and pushes
      // it onto the stack.
      lookupOnContext: function(parts, falsy, scoped) {
        /*jshint -W083 */
        var i = 0,
            len = parts.length;

        if (!scoped && this.options.compat && !this.lastContext) {
          // The depthed query is expected to handle the undefined logic for the root level that
          // is implemented below, so we evaluate that directly in compat mode
          this.push(this.depthedLookup(parts[i++]));
        } else {
          this.pushContext();
        }

        for (; i < len; i++) {
          this.replaceStack(function(current) {
            var lookup = this.nameLookup(current, parts[i], 'context');
            // We want to ensure that zero and false are handled properly if the context (falsy flag)
            // needs to have the special handling for these values.
            if (!falsy) {
              return ' != null ? ' + lookup + ' : ' + current;
            } else {
              // Otherwise we can use generic falsy handling
              return ' && ' + lookup;
            }
          });
        }
      },

      // [lookupData]
      //
      // On stack, before: ...
      // On stack, after: data, ...
      //
      // Push the data lookup operator
      lookupData: function(depth, parts) {
        /*jshint -W083 */
        if (!depth) {
          this.pushStackLiteral('data');
        } else {
          this.pushStackLiteral('this.data(data, ' + depth + ')');
        }

        var len = parts.length;
        for (var i = 0; i < len; i++) {
          this.replaceStack(function(current) {
            return ' && ' + this.nameLookup(current, parts[i], 'data');
          });
        }
      },

      // [resolvePossibleLambda]
      //
      // On stack, before: value, ...
      // On stack, after: resolved value, ...
      //
      // If the `value` is a lambda, replace it on the stack by
      // the return value of the lambda
      resolvePossibleLambda: function() {
        this.aliases.lambda = 'this.lambda';

        this.push('lambda(' + this.popStack() + ', ' + this.contextName(0) + ')');
      },

      // [pushStringParam]
      //
      // On stack, before: ...
      // On stack, after: string, currentContext, ...
      //
      // This opcode is designed for use in string mode, which
      // provides the string value of a parameter along with its
      // depth rather than resolving it immediately.
      pushStringParam: function(string, type) {
        this.pushContext();
        this.pushString(type);

        // If it's a subexpression, the string result
        // will be pushed after this opcode.
        if (type !== 'sexpr') {
          if (typeof string === 'string') {
            this.pushString(string);
          } else {
            this.pushStackLiteral(string);
          }
        }
      },

      emptyHash: function() {
        this.pushStackLiteral('{}');

        if (this.trackIds) {
          this.push('{}'); // hashIds
        }
        if (this.stringParams) {
          this.push('{}'); // hashContexts
          this.push('{}'); // hashTypes
        }
      },
      pushHash: function() {
        if (this.hash) {
          this.hashes.push(this.hash);
        }
        this.hash = {values: [], types: [], contexts: [], ids: []};
      },
      popHash: function() {
        var hash = this.hash;
        this.hash = this.hashes.pop();

        if (this.trackIds) {
          this.push('{' + hash.ids.join(',') + '}');
        }
        if (this.stringParams) {
          this.push('{' + hash.contexts.join(',') + '}');
          this.push('{' + hash.types.join(',') + '}');
        }

        this.push('{\n    ' + hash.values.join(',\n    ') + '\n  }');
      },

      // [pushString]
      //
      // On stack, before: ...
      // On stack, after: quotedString(string), ...
      //
      // Push a quoted version of `string` onto the stack
      pushString: function(string) {
        this.pushStackLiteral(this.quotedString(string));
      },

      // [push]
      //
      // On stack, before: ...
      // On stack, after: expr, ...
      //
      // Push an expression onto the stack
      push: function(expr) {
        this.inlineStack.push(expr);
        return expr;
      },

      // [pushLiteral]
      //
      // On stack, before: ...
      // On stack, after: value, ...
      //
      // Pushes a value onto the stack. This operation prevents
      // the compiler from creating a temporary variable to hold
      // it.
      pushLiteral: function(value) {
        this.pushStackLiteral(value);
      },

      // [pushProgram]
      //
      // On stack, before: ...
      // On stack, after: program(guid), ...
      //
      // Push a program expression onto the stack. This takes
      // a compile-time guid and converts it into a runtime-accessible
      // expression.
      pushProgram: function(guid) {
        if (guid != null) {
          this.pushStackLiteral(this.programExpression(guid));
        } else {
          this.pushStackLiteral(null);
        }
      },

      // [invokeHelper]
      //
      // On stack, before: hash, inverse, program, params..., ...
      // On stack, after: result of helper invocation
      //
      // Pops off the helper's parameters, invokes the helper,
      // and pushes the helper's return value onto the stack.
      //
      // If the helper is not found, `helperMissing` is called.
      invokeHelper: function(paramSize, name, isSimple) {
        this.aliases.helperMissing = 'helpers.helperMissing';

        var nonHelper = this.popStack();
        var helper = this.setupHelper(paramSize, name);

        var lookup = (isSimple ? helper.name + ' || ' : '') + nonHelper + ' || helperMissing';
        this.push('((' + lookup + ').call(' + helper.callParams + '))');
      },

      // [invokeKnownHelper]
      //
      // On stack, before: hash, inverse, program, params..., ...
      // On stack, after: result of helper invocation
      //
      // This operation is used when the helper is known to exist,
      // so a `helperMissing` fallback is not required.
      invokeKnownHelper: function(paramSize, name) {
        var helper = this.setupHelper(paramSize, name);
        this.push(helper.name + ".call(" + helper.callParams + ")");
      },

      // [invokeAmbiguous]
      //
      // On stack, before: hash, inverse, program, params..., ...
      // On stack, after: result of disambiguation
      //
      // This operation is used when an expression like `{{foo}}`
      // is provided, but we don't know at compile-time whether it
      // is a helper or a path.
      //
      // This operation emits more code than the other options,
      // and can be avoided by passing the `knownHelpers` and
      // `knownHelpersOnly` flags at compile-time.
      invokeAmbiguous: function(name, helperCall) {
        this.aliases.functionType = '"function"';
        this.aliases.helperMissing = 'helpers.helperMissing';
        this.useRegister('helper');

        var nonHelper = this.popStack();

        this.emptyHash();
        var helper = this.setupHelper(0, name, helperCall);

        var helperName = this.lastHelper = this.nameLookup('helpers', name, 'helper');

        this.push(
          '((helper = (helper = ' + helperName + ' || ' + nonHelper + ') != null ? helper : helperMissing'
            + (helper.paramsInit ? '),(' + helper.paramsInit : '') + '),'
          + '(typeof helper === functionType ? helper.call(' + helper.callParams + ') : helper))');
      },

      // [invokePartial]
      //
      // On stack, before: context, ...
      // On stack after: result of partial invocation
      //
      // This operation pops off a context, invokes a partial with that context,
      // and pushes the result of the invocation back.
      invokePartial: function(name, indent) {
        var params = [this.nameLookup('partials', name, 'partial'), "'" + indent + "'", "'" + name + "'", this.popStack(), this.popStack(), "helpers", "partials"];

        if (this.options.data) {
          params.push("data");
        } else if (this.options.compat) {
          params.push('undefined');
        }
        if (this.options.compat) {
          params.push('depths');
        }

        this.push("this.invokePartial(" + params.join(", ") + ")");
      },

      // [assignToHash]
      //
      // On stack, before: value, ..., hash, ...
      // On stack, after: ..., hash, ...
      //
      // Pops a value off the stack and assigns it to the current hash
      assignToHash: function(key) {
        var value = this.popStack(),
            context,
            type,
            id;

        if (this.trackIds) {
          id = this.popStack();
        }
        if (this.stringParams) {
          type = this.popStack();
          context = this.popStack();
        }

        var hash = this.hash;
        if (context) {
          hash.contexts.push("'" + key + "': " + context);
        }
        if (type) {
          hash.types.push("'" + key + "': " + type);
        }
        if (id) {
          hash.ids.push("'" + key + "': " + id);
        }
        hash.values.push("'" + key + "': (" + value + ")");
      },

      pushId: function(type, name) {
        if (type === 'ID' || type === 'DATA') {
          this.pushString(name);
        } else if (type === 'sexpr') {
          this.pushStackLiteral('true');
        } else {
          this.pushStackLiteral('null');
        }
      },

      // HELPERS

      compiler: JavaScriptCompiler,

      compileChildren: function(environment, options) {
        var children = environment.children, child, compiler;

        for(var i=0, l=children.length; i<l; i++) {
          child = children[i];
          compiler = new this.compiler();

          var index = this.matchExistingProgram(child);

          if (index == null) {
            this.context.programs.push('');     // Placeholder to prevent name conflicts for nested children
            index = this.context.programs.length;
            child.index = index;
            child.name = 'program' + index;
            this.context.programs[index] = compiler.compile(child, options, this.context, !this.precompile);
            this.context.environments[index] = child;

            this.useDepths = this.useDepths || compiler.useDepths;
          } else {
            child.index = index;
            child.name = 'program' + index;
          }
        }
      },
      matchExistingProgram: function(child) {
        for (var i = 0, len = this.context.environments.length; i < len; i++) {
          var environment = this.context.environments[i];
          if (environment && environment.equals(child)) {
            return i;
          }
        }
      },

      programExpression: function(guid) {
        var child = this.environment.children[guid],
            depths = child.depths.list,
            useDepths = this.useDepths,
            depth;

        var programParams = [child.index, 'data'];

        if (useDepths) {
          programParams.push('depths');
        }

        return 'this.program(' + programParams.join(', ') + ')';
      },

      useRegister: function(name) {
        if(!this.registers[name]) {
          this.registers[name] = true;
          this.registers.list.push(name);
        }
      },

      pushStackLiteral: function(item) {
        return this.push(new Literal(item));
      },

      pushSource: function(source) {
        if (this.pendingContent) {
          this.source.push(this.appendToBuffer(this.quotedString(this.pendingContent)));
          this.pendingContent = undefined;
        }

        if (source) {
          this.source.push(source);
        }
      },

      pushStack: function(item) {
        this.flushInline();

        var stack = this.incrStack();
        this.pushSource(stack + " = " + item + ";");
        this.compileStack.push(stack);
        return stack;
      },

      replaceStack: function(callback) {
        var prefix = '',
            inline = this.isInline(),
            stack,
            createdStack,
            usedLiteral;

        /* istanbul ignore next */
        if (!this.isInline()) {
          throw new Exception('replaceStack on non-inline');
        }

        // We want to merge the inline statement into the replacement statement via ','
        var top = this.popStack(true);

        if (top instanceof Literal) {
          // Literals do not need to be inlined
          prefix = stack = top.value;
          usedLiteral = true;
        } else {
          // Get or create the current stack name for use by the inline
          createdStack = !this.stackSlot;
          var name = !createdStack ? this.topStackName() : this.incrStack();

          prefix = '(' + this.push(name) + ' = ' + top + ')';
          stack = this.topStack();
        }

        var item = callback.call(this, stack);

        if (!usedLiteral) {
          this.popStack();
        }
        if (createdStack) {
          this.stackSlot--;
        }
        this.push('(' + prefix + item + ')');
      },

      incrStack: function() {
        this.stackSlot++;
        if(this.stackSlot > this.stackVars.length) { this.stackVars.push("stack" + this.stackSlot); }
        return this.topStackName();
      },
      topStackName: function() {
        return "stack" + this.stackSlot;
      },
      flushInline: function() {
        var inlineStack = this.inlineStack;
        if (inlineStack.length) {
          this.inlineStack = [];
          for (var i = 0, len = inlineStack.length; i < len; i++) {
            var entry = inlineStack[i];
            if (entry instanceof Literal) {
              this.compileStack.push(entry);
            } else {
              this.pushStack(entry);
            }
          }
        }
      },
      isInline: function() {
        return this.inlineStack.length;
      },

      popStack: function(wrapped) {
        var inline = this.isInline(),
            item = (inline ? this.inlineStack : this.compileStack).pop();

        if (!wrapped && (item instanceof Literal)) {
          return item.value;
        } else {
          if (!inline) {
            /* istanbul ignore next */
            if (!this.stackSlot) {
              throw new Exception('Invalid stack pop');
            }
            this.stackSlot--;
          }
          return item;
        }
      },

      topStack: function() {
        var stack = (this.isInline() ? this.inlineStack : this.compileStack),
            item = stack[stack.length - 1];

        if (item instanceof Literal) {
          return item.value;
        } else {
          return item;
        }
      },

      contextName: function(context) {
        if (this.useDepths && context) {
          return 'depths[' + context + ']';
        } else {
          return 'depth' + context;
        }
      },

      quotedString: function(str) {
        return '"' + str
          .replace(/\\/g, '\\\\')
          .replace(/"/g, '\\"')
          .replace(/\n/g, '\\n')
          .replace(/\r/g, '\\r')
          .replace(/\u2028/g, '\\u2028')   // Per Ecma-262 7.3 + 7.8.4
          .replace(/\u2029/g, '\\u2029') + '"';
      },

      objectLiteral: function(obj) {
        var pairs = [];

        for (var key in obj) {
          if (obj.hasOwnProperty(key)) {
            pairs.push(this.quotedString(key) + ':' + obj[key]);
          }
        }

        return '{' + pairs.join(',') + '}';
      },

      setupHelper: function(paramSize, name, blockHelper) {
        var params = [],
            paramsInit = this.setupParams(name, paramSize, params, blockHelper);
        var foundHelper = this.nameLookup('helpers', name, 'helper');

        return {
          params: params,
          paramsInit: paramsInit,
          name: foundHelper,
          callParams: [this.contextName(0)].concat(params).join(", ")
        };
      },

      setupOptions: function(helper, paramSize, params) {
        var options = {}, contexts = [], types = [], ids = [], param, inverse, program;

        options.name = this.quotedString(helper);
        options.hash = this.popStack();

        if (this.trackIds) {
          options.hashIds = this.popStack();
        }
        if (this.stringParams) {
          options.hashTypes = this.popStack();
          options.hashContexts = this.popStack();
        }

        inverse = this.popStack();
        program = this.popStack();

        // Avoid setting fn and inverse if neither are set. This allows
        // helpers to do a check for `if (options.fn)`
        if (program || inverse) {
          if (!program) {
            program = 'this.noop';
          }

          if (!inverse) {
            inverse = 'this.noop';
          }

          options.fn = program;
          options.inverse = inverse;
        }

        // The parameters go on to the stack in order (making sure that they are evaluated in order)
        // so we need to pop them off the stack in reverse order
        var i = paramSize;
        while (i--) {
          param = this.popStack();
          params[i] = param;

          if (this.trackIds) {
            ids[i] = this.popStack();
          }
          if (this.stringParams) {
            types[i] = this.popStack();
            contexts[i] = this.popStack();
          }
        }

        if (this.trackIds) {
          options.ids = "[" + ids.join(",") + "]";
        }
        if (this.stringParams) {
          options.types = "[" + types.join(",") + "]";
          options.contexts = "[" + contexts.join(",") + "]";
        }

        if (this.options.data) {
          options.data = "data";
        }

        return options;
      },

      // the params and contexts arguments are passed in arrays
      // to fill in
      setupParams: function(helperName, paramSize, params, useRegister) {
        var options = this.objectLiteral(this.setupOptions(helperName, paramSize, params));

        if (useRegister) {
          this.useRegister('options');
          params.push('options');
          return 'options=' + options;
        } else {
          params.push(options);
          return '';
        }
      }
    };

    var reservedWords = (
      "break else new var" +
      " case finally return void" +
      " catch for switch while" +
      " continue function this with" +
      " default if throw" +
      " delete in try" +
      " do instanceof typeof" +
      " abstract enum int short" +
      " boolean export interface static" +
      " byte extends long super" +
      " char final native synchronized" +
      " class float package throws" +
      " const goto private transient" +
      " debugger implements protected volatile" +
      " double import public let yield"
    ).split(" ");

    var compilerWords = JavaScriptCompiler.RESERVED_WORDS = {};

    for(var i=0, l=reservedWords.length; i<l; i++) {
      compilerWords[reservedWords[i]] = true;
    }

    JavaScriptCompiler.isValidJavaScriptVariableName = function(name) {
      return !JavaScriptCompiler.RESERVED_WORDS[name] && /^[a-zA-Z_$][0-9a-zA-Z_$]*$/.test(name);
    };

    __exports__["default"] = JavaScriptCompiler;
  });