/*!

 handlebars v2.0.0

Copyright (C) 2011-2014 by Yehuda Katz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

@license
*/

define(
  'handlebars/safe-string',["exports"],
  function(__exports__) {
    
    // Build out our basic SafeString type
    function SafeString(string) {
      this.string = string;
    }

    SafeString.prototype.toString = function() {
      return "" + this.string;
    };

    __exports__["default"] = SafeString;
  });
define(
  'handlebars/utils',["./safe-string","exports"],
  function(__dependency1__, __exports__) {
    
    /*jshint -W004 */
    var SafeString = __dependency1__["default"];

    var escape = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#x27;",
      "`": "&#x60;"
    };

    var badChars = /[&<>"'`]/g;
    var possible = /[&<>"'`]/;

    function escapeChar(chr) {
      return escape[chr];
    }

    function extend(obj /* , ...source */) {
      for (var i = 1; i < arguments.length; i++) {
        for (var key in arguments[i]) {
          if (Object.prototype.hasOwnProperty.call(arguments[i], key)) {
            obj[key] = arguments[i][key];
          }
        }
      }

      return obj;
    }

    __exports__.extend = extend;var toString = Object.prototype.toString;
    __exports__.toString = toString;
    // Sourced from lodash
    // https://github.com/bestiejs/lodash/blob/master/LICENSE.txt
    var isFunction = function(value) {
      return typeof value === 'function';
    };
    // fallback for older versions of Chrome and Safari
    /* istanbul ignore next */
    if (isFunction(/x/)) {
      isFunction = function(value) {
        return typeof value === 'function' && toString.call(value) === '[object Function]';
      };
    }
    var isFunction;
    __exports__.isFunction = isFunction;
    /* istanbul ignore next */
    var isArray = Array.isArray || function(value) {
      return (value && typeof value === 'object') ? toString.call(value) === '[object Array]' : false;
    };
    __exports__.isArray = isArray;

    function escapeExpression(string) {
      // don't escape SafeStrings, since they're already safe
      if (string instanceof SafeString) {
        return string.toString();
      } else if (string == null) {
        return "";
      } else if (!string) {
        return string + '';
      }

      // Force a string conversion as this will be done by the append regardless and
      // the regex test will do this transparently behind the scenes, causing issues if
      // an object's to string has escaped characters in it.
      string = "" + string;

      if(!possible.test(string)) { return string; }
      return string.replace(badChars, escapeChar);
    }

    __exports__.escapeExpression = escapeExpression;function isEmpty(value) {
      if (!value && value !== 0) {
        return true;
      } else if (isArray(value) && value.length === 0) {
        return true;
      } else {
        return false;
      }
    }

    __exports__.isEmpty = isEmpty;function appendContextPath(contextPath, id) {
      return (contextPath ? contextPath + '.' : '') + id;
    }

    __exports__.appendContextPath = appendContextPath;
  });
define(
  'handlebars/exception',["exports"],
  function(__exports__) {
    

    var errorProps = ['description', 'fileName', 'lineNumber', 'message', 'name', 'number', 'stack'];

    function Exception(message, node) {
      var line;
      if (node && node.firstLine) {
        line = node.firstLine;

        message += ' - ' + line + ':' + node.firstColumn;
      }

      var tmp = Error.prototype.constructor.call(this, message);

      // Unfortunately errors are not enumerable in Chrome (at least), so `for prop in tmp` doesn't work.
      for (var idx = 0; idx < errorProps.length; idx++) {
        this[errorProps[idx]] = tmp[errorProps[idx]];
      }

      if (line) {
        this.lineNumber = line;
        this.column = node.firstColumn;
      }
    }

    Exception.prototype = new Error();

    __exports__["default"] = Exception;
  });
define(
  'handlebars/base',["./utils","./exception","exports"],
  function(__dependency1__, __dependency2__, __exports__) {
    
    var Utils = __dependency1__;
    var Exception = __dependency2__["default"];

    var VERSION = "2.0.0";
    __exports__.VERSION = VERSION;var COMPILER_REVISION = 6;
    __exports__.COMPILER_REVISION = COMPILER_REVISION;
    var REVISION_CHANGES = {
      1: '<= 1.0.rc.2', // 1.0.rc.2 is actually rev2 but doesn't report it
      2: '== 1.0.0-rc.3',
      3: '== 1.0.0-rc.4',
      4: '== 1.x.x',
      5: '== 2.0.0-alpha.x',
      6: '>= 2.0.0-beta.1'
    };
    __exports__.REVISION_CHANGES = REVISION_CHANGES;
    var isArray = Utils.isArray,
        isFunction = Utils.isFunction,
        toString = Utils.toString,
        objectType = '[object Object]';

    function HandlebarsEnvironment(helpers, partials) {
      this.helpers = helpers || {};
      this.partials = partials || {};

      registerDefaultHelpers(this);
    }

    __exports__.HandlebarsEnvironment = HandlebarsEnvironment;HandlebarsEnvironment.prototype = {
      constructor: HandlebarsEnvironment,

      logger: logger,
      log: log,

      registerHelper: function(name, fn) {
        if (toString.call(name) === objectType) {
          if (fn) { throw new Exception('Arg not supported with multiple helpers'); }
          Utils.extend(this.helpers, name);
        } else {
          this.helpers[name] = fn;
        }
      },
      unregisterHelper: function(name) {
        delete this.helpers[name];
      },

      registerPartial: function(name, partial) {
        if (toString.call(name) === objectType) {
          Utils.extend(this.partials,  name);
        } else {
          this.partials[name] = partial;
        }
      },
      unregisterPartial: function(name) {
        delete this.partials[name];
      }
    };

    function registerDefaultHelpers(instance) {
      instance.registerHelper('helperMissing', function(/* [args, ]options */) {
        if(arguments.length === 1) {
          // A missing field in a {{foo}} constuct.
          return undefined;
        } else {
          // Someone is actually trying to call something, blow up.
          throw new Exception("Missing helper: '" + arguments[arguments.length-1].name + "'");
        }
      });

      instance.registerHelper('blockHelperMissing', function(context, options) {
        var inverse = options.inverse,
            fn = options.fn;

        if(context === true) {
          return fn(this);
        } else if(context === false || context == null) {
          return inverse(this);
        } else if (isArray(context)) {
          if(context.length > 0) {
            if (options.ids) {
              options.ids = [options.name];
            }

            return instance.helpers.each(context, options);
          } else {
            return inverse(this);
          }
        } else {
          if (options.data && options.ids) {
            var data = createFrame(options.data);
            data.contextPath = Utils.appendContextPath(options.data.contextPath, options.name);
            options = {data: data};
          }

          return fn(context, options);
        }
      });

      instance.registerHelper('each', function(context, options) {
        if (!options) {
          throw new Exception('Must pass iterator to #each');
        }

        var fn = options.fn, inverse = options.inverse;
        var i = 0, ret = "", data;

        var contextPath;
        if (options.data && options.ids) {
          contextPath = Utils.appendContextPath(options.data.contextPath, options.ids[0]) + '.';
        }

        if (isFunction(context)) { context = context.call(this); }

        if (options.data) {
          data = createFrame(options.data);
        }

        if(context && typeof context === 'object') {
          if (isArray(context)) {
            for(var j = context.length; i<j; i++) {
              if (data) {
                data.index = i;
                data.first = (i === 0);
                data.last  = (i === (context.length-1));

                if (contextPath) {
                  data.contextPath = contextPath + i;
                }
              }
              ret = ret + fn(context[i], { data: data });
            }
          } else {
            for(var key in context) {
              if(context.hasOwnProperty(key)) {
                if(data) {
                  data.key = key;
                  data.index = i;
                  data.first = (i === 0);

                  if (contextPath) {
                    data.contextPath = contextPath + key;
                  }
                }
                ret = ret + fn(context[key], {data: data});
                i++;
              }
            }
          }
        }

        if(i === 0){
          ret = inverse(this);
        }

        return ret;
      });

      instance.registerHelper('if', function(conditional, options) {
        if (isFunction(conditional)) { conditional = conditional.call(this); }

        // Default behavior is to render the positive path if the value is truthy and not empty.
        // The `includeZero` option may be set to treat the condtional as purely not empty based on the
        // behavior of isEmpty. Effectively this determines if 0 is handled by the positive path or negative.
        if ((!options.hash.includeZero && !conditional) || Utils.isEmpty(conditional)) {
          return options.inverse(this);
        } else {
          return options.fn(this);
        }
      });

      instance.registerHelper('unless', function(conditional, options) {
        return instance.helpers['if'].call(this, conditional, {fn: options.inverse, inverse: options.fn, hash: options.hash});
      });

      instance.registerHelper('with', function(context, options) {
        if (isFunction(context)) { context = context.call(this); }

        var fn = options.fn;

        if (!Utils.isEmpty(context)) {
          if (options.data && options.ids) {
            var data = createFrame(options.data);
            data.contextPath = Utils.appendContextPath(options.data.contextPath, options.ids[0]);
            options = {data:data};
          }

          return fn(context, options);
        } else {
          return options.inverse(this);
        }
      });

      instance.registerHelper('log', function(message, options) {
        var level = options.data && options.data.level != null ? parseInt(options.data.level, 10) : 1;
        instance.log(level, message);
      });

      instance.registerHelper('lookup', function(obj, field) {
        return obj && obj[field];
      });
    }

    var logger = {
      methodMap: { 0: 'debug', 1: 'info', 2: 'warn', 3: 'error' },

      // State enum
      DEBUG: 0,
      INFO: 1,
      WARN: 2,
      ERROR: 3,
      level: 3,

      // can be overridden in the host environment
      log: function(level, message) {
        if (logger.level <= level) {
          var method = logger.methodMap[level];
          if (typeof console !== 'undefined' && console[method]) {
            console[method].call(console, message);
          }
        }
      }
    };
    __exports__.logger = logger;
    var log = logger.log;
    __exports__.log = log;
    var createFrame = function(object) {
      var frame = Utils.extend({}, object);
      frame._parent = object;
      return frame;
    };
    __exports__.createFrame = createFrame;
  });
define(
  'handlebars/runtime',["./utils","./exception","./base","exports"],
  function(__dependency1__, __dependency2__, __dependency3__, __exports__) {
    
    var Utils = __dependency1__;
    var Exception = __dependency2__["default"];
    var COMPILER_REVISION = __dependency3__.COMPILER_REVISION;
    var REVISION_CHANGES = __dependency3__.REVISION_CHANGES;
    var createFrame = __dependency3__.createFrame;

    function checkRevision(compilerInfo) {
      var compilerRevision = compilerInfo && compilerInfo[0] || 1,
          currentRevision = COMPILER_REVISION;

      if (compilerRevision !== currentRevision) {
        if (compilerRevision < currentRevision) {
          var runtimeVersions = REVISION_CHANGES[currentRevision],
              compilerVersions = REVISION_CHANGES[compilerRevision];
          throw new Exception("Template was precompiled with an older version of Handlebars than the current runtime. "+
                "Please update your precompiler to a newer version ("+runtimeVersions+") or downgrade your runtime to an older version ("+compilerVersions+").");
        } else {
          // Use the embedded version info since the runtime doesn't know about this revision yet
          throw new Exception("Template was precompiled with a newer version of Handlebars than the current runtime. "+
                "Please update your runtime to a newer version ("+compilerInfo[1]+").");
        }
      }
    }

    __exports__.checkRevision = checkRevision;// TODO: Remove this line and break up compilePartial

    function template(templateSpec, env) {
      /* istanbul ignore next */
      if (!env) {
        throw new Exception("No environment passed to template");
      }
      if (!templateSpec || !templateSpec.main) {
        throw new Exception('Unknown template object: ' + typeof templateSpec);
      }

      // Note: Using env.VM references rather than local var references throughout this section to allow
      // for external users to override these as psuedo-supported APIs.
      env.VM.checkRevision(templateSpec.compiler);

      var invokePartialWrapper = function(partial, indent, name, context, hash, helpers, partials, data, depths) {
        if (hash) {
          context = Utils.extend({}, context, hash);
        }

        var result = env.VM.invokePartial.call(this, partial, name, context, helpers, partials, data, depths);

        if (result == null && env.compile) {
          var options = { helpers: helpers, partials: partials, data: data, depths: depths };
          partials[name] = env.compile(partial, { data: data !== undefined, compat: templateSpec.compat }, env);
          result = partials[name](context, options);
        }
        if (result != null) {
          if (indent) {
            var lines = result.split('\n');
            for (var i = 0, l = lines.length; i < l; i++) {
              if (!lines[i] && i + 1 === l) {
                break;
              }

              lines[i] = indent + lines[i];
            }
            result = lines.join('\n');
          }
          return result;
        } else {
          throw new Exception("The partial " + name + " could not be compiled when running in runtime-only mode");
        }
      };

      // Just add water
      var container = {
        lookup: function(depths, name) {
          var len = depths.length;
          for (var i = 0; i < len; i++) {
            if (depths[i] && depths[i][name] != null) {
              return depths[i][name];
            }
          }
        },
        lambda: function(current, context) {
          return typeof current === 'function' ? current.call(context) : current;
        },

        escapeExpression: Utils.escapeExpression,
        invokePartial: invokePartialWrapper,

        fn: function(i) {
          return templateSpec[i];
        },

        programs: [],
        program: function(i, data, depths) {
          var programWrapper = this.programs[i],
              fn = this.fn(i);
          if (data || depths) {
            programWrapper = program(this, i, fn, data, depths);
          } else if (!programWrapper) {
            programWrapper = this.programs[i] = program(this, i, fn);
          }
          return programWrapper;
        },

        data: function(data, depth) {
          while (data && depth--) {
            data = data._parent;
          }
          return data;
        },
        merge: function(param, common) {
          var ret = param || common;

          if (param && common && (param !== common)) {
            ret = Utils.extend({}, common, param);
          }

          return ret;
        },

        noop: env.VM.noop,
        compilerInfo: templateSpec.compiler
      };

      var ret = function(context, options) {
        options = options || {};
        var data = options.data;

        ret._setup(options);
        if (!options.partial && templateSpec.useData) {
          data = initData(context, data);
        }
        var depths;
        if (templateSpec.useDepths) {
          depths = options.depths ? [context].concat(options.depths) : [context];
        }

        return templateSpec.main.call(container, context, container.helpers, container.partials, data, depths);
      };
      ret.isTop = true;

      ret._setup = function(options) {
        if (!options.partial) {
          container.helpers = container.merge(options.helpers, env.helpers);

          if (templateSpec.usePartial) {
            container.partials = container.merge(options.partials, env.partials);
          }
        } else {
          container.helpers = options.helpers;
          container.partials = options.partials;
        }
      };

      ret._child = function(i, data, depths) {
        if (templateSpec.useDepths && !depths) {
          throw new Exception('must pass parent depths');
        }

        return program(container, i, templateSpec[i], data, depths);
      };
      return ret;
    }

    __exports__.template = template;function program(container, i, fn, data, depths) {
      var prog = function(context, options) {
        options = options || {};

        return fn.call(container, context, container.helpers, container.partials, options.data || data, depths && [context].concat(depths));
      };
      prog.program = i;
      prog.depth = depths ? depths.length : 0;
      return prog;
    }

    __exports__.program = program;function invokePartial(partial, name, context, helpers, partials, data, depths) {
      var options = { partial: true, helpers: helpers, partials: partials, data: data, depths: depths };

      if(partial === undefined) {
        throw new Exception("The partial " + name + " could not be found");
      } else if(partial instanceof Function) {
        return partial(context, options);
      }
    }

    __exports__.invokePartial = invokePartial;function noop() { return ""; }

    __exports__.noop = noop;function initData(context, data) {
      if (!data || !('root' in data)) {
        data = data ? createFrame(data) : {};
        data.root = context;
      }
      return data;
    }
  });
define(
  'handlebars.runtime',["./handlebars/base","./handlebars/safe-string","./handlebars/exception","./handlebars/utils","./handlebars/runtime","exports"],
  function(__dependency1__, __dependency2__, __dependency3__, __dependency4__, __dependency5__, __exports__) {
    
    /*globals Handlebars: true */
    var base = __dependency1__;

    // Each of these augment the Handlebars object. No need to setup here.
    // (This is done to easily share code between commonjs and browse envs)
    var SafeString = __dependency2__["default"];
    var Exception = __dependency3__["default"];
    var Utils = __dependency4__;
    var runtime = __dependency5__;

    // For compatibility and usage outside of module systems, make the Handlebars object a namespace
    var create = function() {
      var hb = new base.HandlebarsEnvironment();

      Utils.extend(hb, base);
      hb.SafeString = SafeString;
      hb.Exception = Exception;
      hb.Utils = Utils;
      hb.escapeExpression = Utils.escapeExpression;

      hb.VM = runtime;
      hb.template = function(spec) {
        return runtime.template(spec, hb);
      };

      return hb;
    };

    var Handlebars = create();
    Handlebars.create = create;

    Handlebars['default'] = Handlebars;

    __exports__["default"] = Handlebars;
  });
define(
  'handlebars/compiler/ast',["../exception","exports"],
  function(__dependency1__, __exports__) {
    
    var Exception = __dependency1__["default"];

    function LocationInfo(locInfo) {
      locInfo = locInfo || {};
      this.firstLine   = locInfo.first_line;
      this.firstColumn = locInfo.first_column;
      this.lastColumn  = locInfo.last_column;
      this.lastLine    = locInfo.last_line;
    }

    var AST = {
      ProgramNode: function(statements, strip, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "program";
        this.statements = statements;
        this.strip = strip;
      },

      MustacheNode: function(rawParams, hash, open, strip, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "mustache";
        this.strip = strip;

        // Open may be a string parsed from the parser or a passed boolean flag
        if (open != null && open.charAt) {
          // Must use charAt to support IE pre-10
          var escapeFlag = open.charAt(3) || open.charAt(2);
          this.escaped = escapeFlag !== '{' && escapeFlag !== '&';
        } else {
          this.escaped = !!open;
        }

        if (rawParams instanceof AST.SexprNode) {
          this.sexpr = rawParams;
        } else {
          // Support old AST API
          this.sexpr = new AST.SexprNode(rawParams, hash);
        }

        // Support old AST API that stored this info in MustacheNode
        this.id = this.sexpr.id;
        this.params = this.sexpr.params;
        this.hash = this.sexpr.hash;
        this.eligibleHelper = this.sexpr.eligibleHelper;
        this.isHelper = this.sexpr.isHelper;
      },

      SexprNode: function(rawParams, hash, locInfo) {
        LocationInfo.call(this, locInfo);

        this.type = "sexpr";
        this.hash = hash;

        var id = this.id = rawParams[0];
        var params = this.params = rawParams.slice(1);

        // a mustache is definitely a helper if:
        // * it is an eligible helper, and
        // * it has at least one parameter or hash segment
        this.isHelper = !!(params.length || hash);

        // a mustache is an eligible helper if:
        // * its id is simple (a single part, not `this` or `..`)
        this.eligibleHelper = this.isHelper || id.isSimple;

        // if a mustache is an eligible helper but not a definite
        // helper, it is ambiguous, and will be resolved in a later
        // pass or at runtime.
      },

      PartialNode: function(partialName, context, hash, strip, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type         = "partial";
        this.partialName  = partialName;
        this.context      = context;
        this.hash = hash;
        this.strip = strip;

        this.strip.inlineStandalone = true;
      },

      BlockNode: function(mustache, program, inverse, strip, locInfo) {
        LocationInfo.call(this, locInfo);

        this.type = 'block';
        this.mustache = mustache;
        this.program  = program;
        this.inverse  = inverse;
        this.strip = strip;

        if (inverse && !program) {
          this.isInverse = true;
        }
      },

      RawBlockNode: function(mustache, content, close, locInfo) {
        LocationInfo.call(this, locInfo);

        if (mustache.sexpr.id.original !== close) {
          throw new Exception(mustache.sexpr.id.original + " doesn't match " + close, this);
        }

        content = new AST.ContentNode(content, locInfo);

        this.type = 'block';
        this.mustache = mustache;
        this.program = new AST.ProgramNode([content], {}, locInfo);
      },

      ContentNode: function(string, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "content";
        this.original = this.string = string;
      },

      HashNode: function(pairs, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "hash";
        this.pairs = pairs;
      },

      IdNode: function(parts, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "ID";

        var original = "",
            dig = [],
            depth = 0,
            depthString = '';

        for(var i=0,l=parts.length; i<l; i++) {
          var part = parts[i].part;
          original += (parts[i].separator || '') + part;

          if (part === ".." || part === "." || part === "this") {
            if (dig.length > 0) {
              throw new Exception("Invalid path: " + original, this);
            } else if (part === "..") {
              depth++;
              depthString += '../';
            } else {
              this.isScoped = true;
            }
          } else {
            dig.push(part);
          }
        }

        this.original = original;
        this.parts    = dig;
        this.string   = dig.join('.');
        this.depth    = depth;
        this.idName   = depthString + this.string;

        // an ID is simple if it only has one part, and that part is not
        // `..` or `this`.
        this.isSimple = parts.length === 1 && !this.isScoped && depth === 0;

        this.stringModeValue = this.string;
      },

      PartialNameNode: function(name, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "PARTIAL_NAME";
        this.name = name.original;
      },

      DataNode: function(id, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "DATA";
        this.id = id;
        this.stringModeValue = id.stringModeValue;
        this.idName = '@' + id.stringModeValue;
      },

      StringNode: function(string, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "STRING";
        this.original =
          this.string =
          this.stringModeValue = string;
      },

      NumberNode: function(number, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "NUMBER";
        this.original =
          this.number = number;
        this.stringModeValue = Number(number);
      },

      BooleanNode: function(bool, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "BOOLEAN";
        this.bool = bool;
        this.stringModeValue = bool === "true";
      },

      CommentNode: function(comment, locInfo) {
        LocationInfo.call(this, locInfo);
        this.type = "comment";
        this.comment = comment;

        this.strip = {
          inlineStandalone: true
        };
      }
    };


    // Must be exported as an object rather than the root of the module as the jison lexer
    // most modify the object to operate properly.
    __exports__["default"] = AST;
  });
define(
  'handlebars/compiler/parser',["exports"],
  function(__exports__) {
    
    /* jshint ignore:start */
    /* istanbul ignore next */
    /* Jison generated parser */
    var handlebars = (function(){
    var parser = {trace: function trace() { },
    yy: {},
    symbols_: {"error":2,"root":3,"program":4,"EOF":5,"program_repetition0":6,"statement":7,"mustache":8,"block":9,"rawBlock":10,"partial":11,"CONTENT":12,"COMMENT":13,"openRawBlock":14,"END_RAW_BLOCK":15,"OPEN_RAW_BLOCK":16,"sexpr":17,"CLOSE_RAW_BLOCK":18,"openBlock":19,"block_option0":20,"closeBlock":21,"openInverse":22,"block_option1":23,"OPEN_BLOCK":24,"CLOSE":25,"OPEN_INVERSE":26,"inverseAndProgram":27,"INVERSE":28,"OPEN_ENDBLOCK":29,"path":30,"OPEN":31,"OPEN_UNESCAPED":32,"CLOSE_UNESCAPED":33,"OPEN_PARTIAL":34,"partialName":35,"param":36,"partial_option0":37,"partial_option1":38,"sexpr_repetition0":39,"sexpr_option0":40,"dataName":41,"STRING":42,"NUMBER":43,"BOOLEAN":44,"OPEN_SEXPR":45,"CLOSE_SEXPR":46,"hash":47,"hash_repetition_plus0":48,"hashSegment":49,"ID":50,"EQUALS":51,"DATA":52,"pathSegments":53,"SEP":54,"$accept":0,"$end":1},
    terminals_: {2:"error",5:"EOF",12:"CONTENT",13:"COMMENT",15:"END_RAW_BLOCK",16:"OPEN_RAW_BLOCK",18:"CLOSE_RAW_BLOCK",24:"OPEN_BLOCK",25:"CLOSE",26:"OPEN_INVERSE",28:"INVERSE",29:"OPEN_ENDBLOCK",31:"OPEN",32:"OPEN_UNESCAPED",33:"CLOSE_UNESCAPED",34:"OPEN_PARTIAL",42:"STRING",43:"NUMBER",44:"BOOLEAN",45:"OPEN_SEXPR",46:"CLOSE_SEXPR",50:"ID",51:"EQUALS",52:"DATA",54:"SEP"},
    productions_: [0,[3,2],[4,1],[7,1],[7,1],[7,1],[7,1],[7,1],[7,1],[10,3],[14,3],[9,4],[9,4],[19,3],[22,3],[27,2],[21,3],[8,3],[8,3],[11,5],[11,4],[17,3],[17,1],[36,1],[36,1],[36,1],[36,1],[36,1],[36,3],[47,1],[49,3],[35,1],[35,1],[35,1],[41,2],[30,1],[53,3],[53,1],[6,0],[6,2],[20,0],[20,1],[23,0],[23,1],[37,0],[37,1],[38,0],[38,1],[39,0],[39,2],[40,0],[40,1],[48,1],[48,2]],
    performAction: function anonymous(yytext,yyleng,yylineno,yy,yystate,$$,_$) {

    var $0 = $$.length - 1;
    switch (yystate) {
    case 1: yy.prepareProgram($$[$0-1].statements, true); return $$[$0-1]; 
    break;
    case 2:this.$ = new yy.ProgramNode(yy.prepareProgram($$[$0]), {}, this._$);
    break;
    case 3:this.$ = $$[$0];
    break;
    case 4:this.$ = $$[$0];
    break;
    case 5:this.$ = $$[$0];
    break;
    case 6:this.$ = $$[$0];
    break;
    case 7:this.$ = new yy.ContentNode($$[$0], this._$);
    break;
    case 8:this.$ = new yy.CommentNode($$[$0], this._$);
    break;
    case 9:this.$ = new yy.RawBlockNode($$[$0-2], $$[$0-1], $$[$0], this._$);
    break;
    case 10:this.$ = new yy.MustacheNode($$[$0-1], null, '', '', this._$);
    break;
    case 11:this.$ = yy.prepareBlock($$[$0-3], $$[$0-2], $$[$0-1], $$[$0], false, this._$);
    break;
    case 12:this.$ = yy.prepareBlock($$[$0-3], $$[$0-2], $$[$0-1], $$[$0], true, this._$);
    break;
    case 13:this.$ = new yy.MustacheNode($$[$0-1], null, $$[$0-2], yy.stripFlags($$[$0-2], $$[$0]), this._$);
    break;
    case 14:this.$ = new yy.MustacheNode($$[$0-1], null, $$[$0-2], yy.stripFlags($$[$0-2], $$[$0]), this._$);
    break;
    case 15:this.$ = { strip: yy.stripFlags($$[$0-1], $$[$0-1]), program: $$[$0] };
    break;
    case 16:this.$ = {path: $$[$0-1], strip: yy.stripFlags($$[$0-2], $$[$0])};
    break;
    case 17:this.$ = new yy.MustacheNode($$[$0-1], null, $$[$0-2], yy.stripFlags($$[$0-2], $$[$0]), this._$);
    break;
    case 18:this.$ = new yy.MustacheNode($$[$0-1], null, $$[$0-2], yy.stripFlags($$[$0-2], $$[$0]), this._$);
    break;
    case 19:this.$ = new yy.PartialNode($$[$0-3], $$[$0-2], $$[$0-1], yy.stripFlags($$[$0-4], $$[$0]), this._$);
    break;
    case 20:this.$ = new yy.PartialNode($$[$0-2], undefined, $$[$0-1], yy.stripFlags($$[$0-3], $$[$0]), this._$);
    break;
    case 21:this.$ = new yy.SexprNode([$$[$0-2]].concat($$[$0-1]), $$[$0], this._$);
    break;
    case 22:this.$ = new yy.SexprNode([$$[$0]], null, this._$);
    break;
    case 23:this.$ = $$[$0];
    break;
    case 24:this.$ = new yy.StringNode($$[$0], this._$);
    break;
    case 25:this.$ = new yy.NumberNode($$[$0], this._$);
    break;
    case 26:this.$ = new yy.BooleanNode($$[$0], this._$);
    break;
    case 27:this.$ = $$[$0];
    break;
    case 28:$$[$0-1].isHelper = true; this.$ = $$[$0-1];
    break;
    case 29:this.$ = new yy.HashNode($$[$0], this._$);
    break;
    case 30:this.$ = [$$[$0-2], $$[$0]];
    break;
    case 31:this.$ = new yy.PartialNameNode($$[$0], this._$);
    break;
    case 32:this.$ = new yy.PartialNameNode(new yy.StringNode($$[$0], this._$), this._$);
    break;
    case 33:this.$ = new yy.PartialNameNode(new yy.NumberNode($$[$0], this._$));
    break;
    case 34:this.$ = new yy.DataNode($$[$0], this._$);
    break;
    case 35:this.$ = new yy.IdNode($$[$0], this._$);
    break;
    case 36: $$[$0-2].push({part: $$[$0], separator: $$[$0-1]}); this.$ = $$[$0-2]; 
    break;
    case 37:this.$ = [{part: $$[$0]}];
    break;
    case 38:this.$ = [];
    break;
    case 39:$$[$0-1].push($$[$0]);
    break;
    case 48:this.$ = [];
    break;
    case 49:$$[$0-1].push($$[$0]);
    break;
    case 52:this.$ = [$$[$0]];
    break;
    case 53:$$[$0-1].push($$[$0]);
    break;
    }
    },
    table: [{3:1,4:2,5:[2,38],6:3,12:[2,38],13:[2,38],16:[2,38],24:[2,38],26:[2,38],31:[2,38],32:[2,38],34:[2,38]},{1:[3]},{5:[1,4]},{5:[2,2],7:5,8:6,9:7,10:8,11:9,12:[1,10],13:[1,11],14:16,16:[1,20],19:14,22:15,24:[1,18],26:[1,19],28:[2,2],29:[2,2],31:[1,12],32:[1,13],34:[1,17]},{1:[2,1]},{5:[2,39],12:[2,39],13:[2,39],16:[2,39],24:[2,39],26:[2,39],28:[2,39],29:[2,39],31:[2,39],32:[2,39],34:[2,39]},{5:[2,3],12:[2,3],13:[2,3],16:[2,3],24:[2,3],26:[2,3],28:[2,3],29:[2,3],31:[2,3],32:[2,3],34:[2,3]},{5:[2,4],12:[2,4],13:[2,4],16:[2,4],24:[2,4],26:[2,4],28:[2,4],29:[2,4],31:[2,4],32:[2,4],34:[2,4]},{5:[2,5],12:[2,5],13:[2,5],16:[2,5],24:[2,5],26:[2,5],28:[2,5],29:[2,5],31:[2,5],32:[2,5],34:[2,5]},{5:[2,6],12:[2,6],13:[2,6],16:[2,6],24:[2,6],26:[2,6],28:[2,6],29:[2,6],31:[2,6],32:[2,6],34:[2,6]},{5:[2,7],12:[2,7],13:[2,7],16:[2,7],24:[2,7],26:[2,7],28:[2,7],29:[2,7],31:[2,7],32:[2,7],34:[2,7]},{5:[2,8],12:[2,8],13:[2,8],16:[2,8],24:[2,8],26:[2,8],28:[2,8],29:[2,8],31:[2,8],32:[2,8],34:[2,8]},{17:21,30:22,41:23,50:[1,26],52:[1,25],53:24},{17:27,30:22,41:23,50:[1,26],52:[1,25],53:24},{4:28,6:3,12:[2,38],13:[2,38],16:[2,38],24:[2,38],26:[2,38],28:[2,38],29:[2,38],31:[2,38],32:[2,38],34:[2,38]},{4:29,6:3,12:[2,38],13:[2,38],16:[2,38],24:[2,38],26:[2,38],28:[2,38],29:[2,38],31:[2,38],32:[2,38],34:[2,38]},{12:[1,30]},{30:32,35:31,42:[1,33],43:[1,34],50:[1,26],53:24},{17:35,30:22,41:23,50:[1,26],52:[1,25],53:24},{17:36,30:22,41:23,50:[1,26],52:[1,25],53:24},{17:37,30:22,41:23,50:[1,26],52:[1,25],53:24},{25:[1,38]},{18:[2,48],25:[2,48],33:[2,48],39:39,42:[2,48],43:[2,48],44:[2,48],45:[2,48],46:[2,48],50:[2,48],52:[2,48]},{18:[2,22],25:[2,22],33:[2,22],46:[2,22]},{18:[2,35],25:[2,35],33:[2,35],42:[2,35],43:[2,35],44:[2,35],45:[2,35],46:[2,35],50:[2,35],52:[2,35],54:[1,40]},{30:41,50:[1,26],53:24},{18:[2,37],25:[2,37],33:[2,37],42:[2,37],43:[2,37],44:[2,37],45:[2,37],46:[2,37],50:[2,37],52:[2,37],54:[2,37]},{33:[1,42]},{20:43,27:44,28:[1,45],29:[2,40]},{23:46,27:47,28:[1,45],29:[2,42]},{15:[1,48]},{25:[2,46],30:51,36:49,38:50,41:55,42:[1,52],43:[1,53],44:[1,54],45:[1,56],47:57,48:58,49:60,50:[1,59],52:[1,25],53:24},{25:[2,31],42:[2,31],43:[2,31],44:[2,31],45:[2,31],50:[2,31],52:[2,31]},{25:[2,32],42:[2,32],43:[2,32],44:[2,32],45:[2,32],50:[2,32],52:[2,32]},{25:[2,33],42:[2,33],43:[2,33],44:[2,33],45:[2,33],50:[2,33],52:[2,33]},{25:[1,61]},{25:[1,62]},{18:[1,63]},{5:[2,17],12:[2,17],13:[2,17],16:[2,17],24:[2,17],26:[2,17],28:[2,17],29:[2,17],31:[2,17],32:[2,17],34:[2,17]},{18:[2,50],25:[2,50],30:51,33:[2,50],36:65,40:64,41:55,42:[1,52],43:[1,53],44:[1,54],45:[1,56],46:[2,50],47:66,48:58,49:60,50:[1,59],52:[1,25],53:24},{50:[1,67]},{18:[2,34],25:[2,34],33:[2,34],42:[2,34],43:[2,34],44:[2,34],45:[2,34],46:[2,34],50:[2,34],52:[2,34]},{5:[2,18],12:[2,18],13:[2,18],16:[2,18],24:[2,18],26:[2,18],28:[2,18],29:[2,18],31:[2,18],32:[2,18],34:[2,18]},{21:68,29:[1,69]},{29:[2,41]},{4:70,6:3,12:[2,38],13:[2,38],16:[2,38],24:[2,38],26:[2,38],29:[2,38],31:[2,38],32:[2,38],34:[2,38]},{21:71,29:[1,69]},{29:[2,43]},{5:[2,9],12:[2,9],13:[2,9],16:[2,9],24:[2,9],26:[2,9],28:[2,9],29:[2,9],31:[2,9],32:[2,9],34:[2,9]},{25:[2,44],37:72,47:73,48:58,49:60,50:[1,74]},{25:[1,75]},{18:[2,23],25:[2,23],33:[2,23],42:[2,23],43:[2,23],44:[2,23],45:[2,23],46:[2,23],50:[2,23],52:[2,23]},{18:[2,24],25:[2,24],33:[2,24],42:[2,24],43:[2,24],44:[2,24],45:[2,24],46:[2,24],50:[2,24],52:[2,24]},{18:[2,25],25:[2,25],33:[2,25],42:[2,25],43:[2,25],44:[2,25],45:[2,25],46:[2,25],50:[2,25],52:[2,25]},{18:[2,26],25:[2,26],33:[2,26],42:[2,26],43:[2,26],44:[2,26],45:[2,26],46:[2,26],50:[2,26],52:[2,26]},{18:[2,27],25:[2,27],33:[2,27],42:[2,27],43:[2,27],44:[2,27],45:[2,27],46:[2,27],50:[2,27],52:[2,27]},{17:76,30:22,41:23,50:[1,26],52:[1,25],53:24},{25:[2,47]},{18:[2,29],25:[2,29],33:[2,29],46:[2,29],49:77,50:[1,74]},{18:[2,37],25:[2,37],33:[2,37],42:[2,37],43:[2,37],44:[2,37],45:[2,37],46:[2,37],50:[2,37],51:[1,78],52:[2,37],54:[2,37]},{18:[2,52],25:[2,52],33:[2,52],46:[2,52],50:[2,52]},{12:[2,13],13:[2,13],16:[2,13],24:[2,13],26:[2,13],28:[2,13],29:[2,13],31:[2,13],32:[2,13],34:[2,13]},{12:[2,14],13:[2,14],16:[2,14],24:[2,14],26:[2,14],28:[2,14],29:[2,14],31:[2,14],32:[2,14],34:[2,14]},{12:[2,10]},{18:[2,21],25:[2,21],33:[2,21],46:[2,21]},{18:[2,49],25:[2,49],33:[2,49],42:[2,49],43:[2,49],44:[2,49],45:[2,49],46:[2,49],50:[2,49],52:[2,49]},{18:[2,51],25:[2,51],33:[2,51],46:[2,51]},{18:[2,36],25:[2,36],33:[2,36],42:[2,36],43:[2,36],44:[2,36],45:[2,36],46:[2,36],50:[2,36],52:[2,36],54:[2,36]},{5:[2,11],12:[2,11],13:[2,11],16:[2,11],24:[2,11],26:[2,11],28:[2,11],29:[2,11],31:[2,11],32:[2,11],34:[2,11]},{30:79,50:[1,26],53:24},{29:[2,15]},{5:[2,12],12:[2,12],13:[2,12],16:[2,12],24:[2,12],26:[2,12],28:[2,12],29:[2,12],31:[2,12],32:[2,12],34:[2,12]},{25:[1,80]},{25:[2,45]},{51:[1,78]},{5:[2,20],12:[2,20],13:[2,20],16:[2,20],24:[2,20],26:[2,20],28:[2,20],29:[2,20],31:[2,20],32:[2,20],34:[2,20]},{46:[1,81]},{18:[2,53],25:[2,53],33:[2,53],46:[2,53],50:[2,53]},{30:51,36:82,41:55,42:[1,52],43:[1,53],44:[1,54],45:[1,56],50:[1,26],52:[1,25],53:24},{25:[1,83]},{5:[2,19],12:[2,19],13:[2,19],16:[2,19],24:[2,19],26:[2,19],28:[2,19],29:[2,19],31:[2,19],32:[2,19],34:[2,19]},{18:[2,28],25:[2,28],33:[2,28],42:[2,28],43:[2,28],44:[2,28],45:[2,28],46:[2,28],50:[2,28],52:[2,28]},{18:[2,30],25:[2,30],33:[2,30],46:[2,30],50:[2,30]},{5:[2,16],12:[2,16],13:[2,16],16:[2,16],24:[2,16],26:[2,16],28:[2,16],29:[2,16],31:[2,16],32:[2,16],34:[2,16]}],
    defaultActions: {4:[2,1],44:[2,41],47:[2,43],57:[2,47],63:[2,10],70:[2,15],73:[2,45]},
    parseError: function parseError(str, hash) {
        throw new Error(str);
    },
    parse: function parse(input) {
        var self = this, stack = [0], vstack = [null], lstack = [], table = this.table, yytext = "", yylineno = 0, yyleng = 0, recovering = 0, TERROR = 2, EOF = 1;
        this.lexer.setInput(input);
        this.lexer.yy = this.yy;
        this.yy.lexer = this.lexer;
        this.yy.parser = this;
        if (typeof this.lexer.yylloc == "undefined")
            this.lexer.yylloc = {};
        var yyloc = this.lexer.yylloc;
        lstack.push(yyloc);
        var ranges = this.lexer.options && this.lexer.options.ranges;
        if (typeof this.yy.parseError === "function")
            this.parseError = this.yy.parseError;
        function popStack(n) {
            stack.length = stack.length - 2 * n;
            vstack.length = vstack.length - n;
            lstack.length = lstack.length - n;
        }
        function lex() {
            var token;
            token = self.lexer.lex() || 1;
            if (typeof token !== "number") {
                token = self.symbols_[token] || token;
            }
            return token;
        }
        var symbol, preErrorSymbol, state, action, a, r, yyval = {}, p, len, newState, expected;
        while (true) {
            state = stack[stack.length - 1];
            if (this.defaultActions[state]) {
                action = this.defaultActions[state];
            } else {
                if (symbol === null || typeof symbol == "undefined") {
                    symbol = lex();
                }
                action = table[state] && table[state][symbol];
            }
            if (typeof action === "undefined" || !action.length || !action[0]) {
                var errStr = "";
                if (!recovering) {
                    expected = [];
                    for (p in table[state])
                        if (this.terminals_[p] && p > 2) {
                            expected.push("'" + this.terminals_[p] + "'");
                        }
                    if (this.lexer.showPosition) {
                        errStr = "Parse error on line " + (yylineno + 1) + ":\n" + this.lexer.showPosition() + "\nExpecting " + expected.join(", ") + ", got '" + (this.terminals_[symbol] || symbol) + "'";
                    } else {
                        errStr = "Parse error on line " + (yylineno + 1) + ": Unexpected " + (symbol == 1?"end of input":"'" + (this.terminals_[symbol] || symbol) + "'");
                    }
                    this.parseError(errStr, {text: this.lexer.match, token: this.terminals_[symbol] || symbol, line: this.lexer.yylineno, loc: yyloc, expected: expected});
                }
            }
            if (action[0] instanceof Array && action.length > 1) {
                throw new Error("Parse Error: multiple actions possible at state: " + state + ", token: " + symbol);
            }
            switch (action[0]) {
            case 1:
                stack.push(symbol);
                vstack.push(this.lexer.yytext);
                lstack.push(this.lexer.yylloc);
                stack.push(action[1]);
                symbol = null;
                if (!preErrorSymbol) {
                    yyleng = this.lexer.yyleng;
                    yytext = this.lexer.yytext;
                    yylineno = this.lexer.yylineno;
                    yyloc = this.lexer.yylloc;
                    if (recovering > 0)
                        recovering--;
                } else {
                    symbol = preErrorSymbol;
                    preErrorSymbol = null;
                }
                break;
            case 2:
                len = this.productions_[action[1]][1];
                yyval.$ = vstack[vstack.length - len];
                yyval._$ = {first_line: lstack[lstack.length - (len || 1)].first_line, last_line: lstack[lstack.length - 1].last_line, first_column: lstack[lstack.length - (len || 1)].first_column, last_column: lstack[lstack.length - 1].last_column};
                if (ranges) {
                    yyval._$.range = [lstack[lstack.length - (len || 1)].range[0], lstack[lstack.length - 1].range[1]];
                }
                r = this.performAction.call(yyval, yytext, yyleng, yylineno, this.yy, action[1], vstack, lstack);
                if (typeof r !== "undefined") {
                    return r;
                }
                if (len) {
                    stack = stack.slice(0, -1 * len * 2);
                    vstack = vstack.slice(0, -1 * len);
                    lstack = lstack.slice(0, -1 * len);
                }
                stack.push(this.productions_[action[1]][0]);
                vstack.push(yyval.$);
                lstack.push(yyval._$);
                newState = table[stack[stack.length - 2]][stack[stack.length - 1]];
                stack.push(newState);
                break;
            case 3:
                return true;
            }
        }
        return true;
    }
    };
    /* Jison generated lexer */
    var lexer = (function(){
    var lexer = ({EOF:1,
    parseError:function parseError(str, hash) {
            if (this.yy.parser) {
                this.yy.parser.parseError(str, hash);
            } else {
                throw new Error(str);
            }
        },
    setInput:function (input) {
            this._input = input;
            this._more = this._less = this.done = false;
            this.yylineno = this.yyleng = 0;
            this.yytext = this.matched = this.match = '';
            this.conditionStack = ['INITIAL'];
            this.yylloc = {first_line:1,first_column:0,last_line:1,last_column:0};
            if (this.options.ranges) this.yylloc.range = [0,0];
            this.offset = 0;
            return this;
        },
    input:function () {
            var ch = this._input[0];
            this.yytext += ch;
            this.yyleng++;
            this.offset++;
            this.match += ch;
            this.matched += ch;
            var lines = ch.match(/(?:\r\n?|\n).*/g);
            if (lines) {
                this.yylineno++;
                this.yylloc.last_line++;
            } else {
                this.yylloc.last_column++;
            }
            if (this.options.ranges) this.yylloc.range[1]++;

            this._input = this._input.slice(1);
            return ch;
        },
    unput:function (ch) {
            var len = ch.length;
            var lines = ch.split(/(?:\r\n?|\n)/g);

            this._input = ch + this._input;
            this.yytext = this.yytext.substr(0, this.yytext.length-len-1);
            //this.yyleng -= len;
            this.offset -= len;
            var oldLines = this.match.split(/(?:\r\n?|\n)/g);
            this.match = this.match.substr(0, this.match.length-1);
            this.matched = this.matched.substr(0, this.matched.length-1);

            if (lines.length-1) this.yylineno -= lines.length-1;
            var r = this.yylloc.range;

            this.yylloc = {first_line: this.yylloc.first_line,
              last_line: this.yylineno+1,
              first_column: this.yylloc.first_column,
              last_column: lines ?
                  (lines.length === oldLines.length ? this.yylloc.first_column : 0) + oldLines[oldLines.length - lines.length].length - lines[0].length:
                  this.yylloc.first_column - len
              };

            if (this.options.ranges) {
                this.yylloc.range = [r[0], r[0] + this.yyleng - len];
            }
            return this;
        },
    more:function () {
            this._more = true;
            return this;
        },
    less:function (n) {
            this.unput(this.match.slice(n));
        },
    pastInput:function () {
            var past = this.matched.substr(0, this.matched.length - this.match.length);
            return (past.length > 20 ? '...':'') + past.substr(-20).replace(/\n/g, "");
        },
    upcomingInput:function () {
            var next = this.match;
            if (next.length < 20) {
                next += this._input.substr(0, 20-next.length);
            }
            return (next.substr(0,20)+(next.length > 20 ? '...':'')).replace(/\n/g, "");
        },
    showPosition:function () {
            var pre = this.pastInput();
            var c = new Array(pre.length + 1).join("-");
            return pre + this.upcomingInput() + "\n" + c+"^";
        },
    next:function () {
            if (this.done) {
                return this.EOF;
            }
            if (!this._input) this.done = true;

            var token,
                match,
                tempMatch,
                index,
                col,
                lines;
            if (!this._more) {
                this.yytext = '';
                this.match = '';
            }
            var rules = this._currentRules();
            for (var i=0;i < rules.length; i++) {
                tempMatch = this._input.match(this.rules[rules[i]]);
                if (tempMatch && (!match || tempMatch[0].length > match[0].length)) {
                    match = tempMatch;
                    index = i;
                    if (!this.options.flex) break;
                }
            }
            if (match) {
                lines = match[0].match(/(?:\r\n?|\n).*/g);
                if (lines) this.yylineno += lines.length;
                this.yylloc = {first_line: this.yylloc.last_line,
                               last_line: this.yylineno+1,
                               first_column: this.yylloc.last_column,
                               last_column: lines ? lines[lines.length-1].length-lines[lines.length-1].match(/\r?\n?/)[0].length : this.yylloc.last_column + match[0].length};
                this.yytext += match[0];
                this.match += match[0];
                this.matches = match;
                this.yyleng = this.yytext.length;
                if (this.options.ranges) {
                    this.yylloc.range = [this.offset, this.offset += this.yyleng];
                }
                this._more = false;
                this._input = this._input.slice(match[0].length);
                this.matched += match[0];
                token = this.performAction.call(this, this.yy, this, rules[index],this.conditionStack[this.conditionStack.length-1]);
                if (this.done && this._input) this.done = false;
                if (token) return token;
                else return;
            }
            if (this._input === "") {
                return this.EOF;
            } else {
                return this.parseError('Lexical error on line '+(this.yylineno+1)+'. Unrecognized text.\n'+this.showPosition(),
                        {text: "", token: null, line: this.yylineno});
            }
        },
    lex:function lex() {
            var r = this.next();
            if (typeof r !== 'undefined') {
                return r;
            } else {
                return this.lex();
            }
        },
    begin:function begin(condition) {
            this.conditionStack.push(condition);
        },
    popState:function popState() {
            return this.conditionStack.pop();
        },
    _currentRules:function _currentRules() {
            return this.conditions[this.conditionStack[this.conditionStack.length-1]].rules;
        },
    topState:function () {
            return this.conditionStack[this.conditionStack.length-2];
        },
    pushState:function begin(condition) {
            this.begin(condition);
        }});
    lexer.options = {};
    lexer.performAction = function anonymous(yy,yy_,$avoiding_name_collisions,YY_START) {


    function strip(start, end) {
      return yy_.yytext = yy_.yytext.substr(start, yy_.yyleng-end);
    }


    var YYSTATE=YY_START
    switch($avoiding_name_collisions) {
    case 0:
                                       if(yy_.yytext.slice(-2) === "\\\\") {
                                         strip(0,1);
                                         this.begin("mu");
                                       } else if(yy_.yytext.slice(-1) === "\\") {
                                         strip(0,1);
                                         this.begin("emu");
                                       } else {
                                         this.begin("mu");
                                       }
                                       if(yy_.yytext) return 12;
                                     
    break;
    case 1:return 12;
    break;
    case 2:
                                       this.popState();
                                       return 12;
                                     
    break;
    case 3:
                                      yy_.yytext = yy_.yytext.substr(5, yy_.yyleng-9);
                                      this.popState();
                                      return 15;
                                     
    break;
    case 4: return 12; 
    break;
    case 5:strip(0,4); this.popState(); return 13;
    break;
    case 6:return 45;
    break;
    case 7:return 46;
    break;
    case 8: return 16; 
    break;
    case 9:
                                      this.popState();
                                      this.begin('raw');
                                      return 18;
                                     
    break;
    case 10:return 34;
    break;
    case 11:return 24;
    break;
    case 12:return 29;
    break;
    case 13:this.popState(); return 28;
    break;
    case 14:this.popState(); return 28;
    break;
    case 15:return 26;
    break;
    case 16:return 26;
    break;
    case 17:return 32;
    break;
    case 18:return 31;
    break;
    case 19:this.popState(); this.begin('com');
    break;
    case 20:strip(3,5); this.popState(); return 13;
    break;
    case 21:return 31;
    break;
    case 22:return 51;
    break;
    case 23:return 50;
    break;
    case 24:return 50;
    break;
    case 25:return 54;
    break;
    case 26:// ignore whitespace
    break;
    case 27:this.popState(); return 33;
    break;
    case 28:this.popState(); return 25;
    break;
    case 29:yy_.yytext = strip(1,2).replace(/\\"/g,'"'); return 42;
    break;
    case 30:yy_.yytext = strip(1,2).replace(/\\'/g,"'"); return 42;
    break;
    case 31:return 52;
    break;
    case 32:return 44;
    break;
    case 33:return 44;
    break;
    case 34:return 43;
    break;
    case 35:return 50;
    break;
    case 36:yy_.yytext = strip(1,2); return 50;
    break;
    case 37:return 'INVALID';
    break;
    case 38:return 5;
    break;
    }
    };
    lexer.rules = [/^(?:[^\x00]*?(?=(\{\{)))/,/^(?:[^\x00]+)/,/^(?:[^\x00]{2,}?(?=(\{\{|\\\{\{|\\\\\{\{|$)))/,/^(?:\{\{\{\{\/[^\s!"#%-,\.\/;->@\[-\^`\{-~]+(?=[=}\s\/.])\}\}\}\})/,/^(?:[^\x00]*?(?=(\{\{\{\{\/)))/,/^(?:[\s\S]*?--\}\})/,/^(?:\()/,/^(?:\))/,/^(?:\{\{\{\{)/,/^(?:\}\}\}\})/,/^(?:\{\{(~)?>)/,/^(?:\{\{(~)?#)/,/^(?:\{\{(~)?\/)/,/^(?:\{\{(~)?\^\s*(~)?\}\})/,/^(?:\{\{(~)?\s*else\s*(~)?\}\})/,/^(?:\{\{(~)?\^)/,/^(?:\{\{(~)?\s*else\b)/,/^(?:\{\{(~)?\{)/,/^(?:\{\{(~)?&)/,/^(?:\{\{!--)/,/^(?:\{\{![\s\S]*?\}\})/,/^(?:\{\{(~)?)/,/^(?:=)/,/^(?:\.\.)/,/^(?:\.(?=([=~}\s\/.)])))/,/^(?:[\/.])/,/^(?:\s+)/,/^(?:\}(~)?\}\})/,/^(?:(~)?\}\})/,/^(?:"(\\["]|[^"])*")/,/^(?:'(\\[']|[^'])*')/,/^(?:@)/,/^(?:true(?=([~}\s)])))/,/^(?:false(?=([~}\s)])))/,/^(?:-?[0-9]+(?:\.[0-9]+)?(?=([~}\s)])))/,/^(?:([^\s!"#%-,\.\/;->@\[-\^`\{-~]+(?=([=~}\s\/.)]))))/,/^(?:\[[^\]]*\])/,/^(?:.)/,/^(?:$)/];
    lexer.conditions = {"mu":{"rules":[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38],"inclusive":false},"emu":{"rules":[2],"inclusive":false},"com":{"rules":[5],"inclusive":false},"raw":{"rules":[3,4],"inclusive":false},"INITIAL":{"rules":[0,1,38],"inclusive":true}};
    return lexer;})()
    parser.lexer = lexer;
    function Parser () { this.yy = {}; }Parser.prototype = parser;parser.Parser = Parser;
    return new Parser;
    })();__exports__["default"] = handlebars;
    /* jshint ignore:end */
  });
define(
  'handlebars/compiler/helpers',["../exception","exports"],
  function(__dependency1__, __exports__) {
    
    var Exception = __dependency1__["default"];

    function stripFlags(open, close) {
      return {
        left: open.charAt(2) === '~',
        right: close.charAt(close.length-3) === '~'
      };
    }

    __exports__.stripFlags = stripFlags;
    function prepareBlock(mustache, program, inverseAndProgram, close, inverted, locInfo) {
      /*jshint -W040 */
      if (mustache.sexpr.id.original !== close.path.original) {
        throw new Exception(mustache.sexpr.id.original + ' doesn\'t match ' + close.path.original, mustache);
      }

      var inverse = inverseAndProgram && inverseAndProgram.program;

      var strip = {
        left: mustache.strip.left,
        right: close.strip.right,

        // Determine the standalone candiacy. Basically flag our content as being possibly standalone
        // so our parent can determine if we actually are standalone
        openStandalone: isNextWhitespace(program.statements),
        closeStandalone: isPrevWhitespace((inverse || program).statements)
      };

      if (mustache.strip.right) {
        omitRight(program.statements, null, true);
      }

      if (inverse) {
        var inverseStrip = inverseAndProgram.strip;

        if (inverseStrip.left) {
          omitLeft(program.statements, null, true);
        }
        if (inverseStrip.right) {
          omitRight(inverse.statements, null, true);
        }
        if (close.strip.left) {
          omitLeft(inverse.statements, null, true);
        }

        // Find standalone else statments
        if (isPrevWhitespace(program.statements)
            && isNextWhitespace(inverse.statements)) {

          omitLeft(program.statements);
          omitRight(inverse.statements);
        }
      } else {
        if (close.strip.left) {
          omitLeft(program.statements, null, true);
        }
      }

      if (inverted) {
        return new this.BlockNode(mustache, inverse, program, strip, locInfo);
      } else {
        return new this.BlockNode(mustache, program, inverse, strip, locInfo);
      }
    }

    __exports__.prepareBlock = prepareBlock;
    function prepareProgram(statements, isRoot) {
      for (var i = 0, l = statements.length; i < l; i++) {
        var current = statements[i],
            strip = current.strip;

        if (!strip) {
          continue;
        }

        var _isPrevWhitespace = isPrevWhitespace(statements, i, isRoot, current.type === 'partial'),
            _isNextWhitespace = isNextWhitespace(statements, i, isRoot),

            openStandalone = strip.openStandalone && _isPrevWhitespace,
            closeStandalone = strip.closeStandalone && _isNextWhitespace,
            inlineStandalone = strip.inlineStandalone && _isPrevWhitespace && _isNextWhitespace;

        if (strip.right) {
          omitRight(statements, i, true);
        }
        if (strip.left) {
          omitLeft(statements, i, true);
        }

        if (inlineStandalone) {
          omitRight(statements, i);

          if (omitLeft(statements, i)) {
            // If we are on a standalone node, save the indent info for partials
            if (current.type === 'partial') {
              current.indent = (/([ \t]+$)/).exec(statements[i-1].original) ? RegExp.$1 : '';
            }
          }
        }
        if (openStandalone) {
          omitRight((current.program || current.inverse).statements);

          // Strip out the previous content node if it's whitespace only
          omitLeft(statements, i);
        }
        if (closeStandalone) {
          // Always strip the next node
          omitRight(statements, i);

          omitLeft((current.inverse || current.program).statements);
        }
      }

      return statements;
    }

    __exports__.prepareProgram = prepareProgram;function isPrevWhitespace(statements, i, isRoot) {
      if (i === undefined) {
        i = statements.length;
      }

      // Nodes that end with newlines are considered whitespace (but are special
      // cased for strip operations)
      var prev = statements[i-1],
          sibling = statements[i-2];
      if (!prev) {
        return isRoot;
      }

      if (prev.type === 'content') {
        return (sibling || !isRoot ? (/\r?\n\s*?$/) : (/(^|\r?\n)\s*?$/)).test(prev.original);
      }
    }
    function isNextWhitespace(statements, i, isRoot) {
      if (i === undefined) {
        i = -1;
      }

      var next = statements[i+1],
          sibling = statements[i+2];
      if (!next) {
        return isRoot;
      }

      if (next.type === 'content') {
        return (sibling || !isRoot ? (/^\s*?\r?\n/) : (/^\s*?(\r?\n|$)/)).test(next.original);
      }
    }

    // Marks the node to the right of the position as omitted.
    // I.e. {{foo}}' ' will mark the ' ' node as omitted.
    //
    // If i is undefined, then the first child will be marked as such.
    //
    // If mulitple is truthy then all whitespace will be stripped out until non-whitespace
    // content is met.
    function omitRight(statements, i, multiple) {
      var current = statements[i == null ? 0 : i + 1];
      if (!current || current.type !== 'content' || (!multiple && current.rightStripped)) {
        return;
      }

      var original = current.string;
      current.string = current.string.replace(multiple ? (/^\s+/) : (/^[ \t]*\r?\n?/), '');
      current.rightStripped = current.string !== original;
    }

    // Marks the node to the left of the position as omitted.
    // I.e. ' '{{foo}} will mark the ' ' node as omitted.
    //
    // If i is undefined then the last child will be marked as such.
    //
    // If mulitple is truthy then all whitespace will be stripped out until non-whitespace
    // content is met.
    function omitLeft(statements, i, multiple) {
      var current = statements[i == null ? statements.length - 1 : i - 1];
      if (!current || current.type !== 'content' || (!multiple && current.leftStripped)) {
        return;
      }

      // We omit the last node if it's whitespace only and not preceeded by a non-content node.
      var original = current.string;
      current.string = current.string.replace(multiple ? (/\s+$/) : (/[ \t]+$/), '');
      current.leftStripped = current.string !== original;
      return current.leftStripped;
    }
  });
define(
  'handlebars/compiler/base',["./parser","./ast","./helpers","../utils","exports"],
  function(__dependency1__, __dependency2__, __dependency3__, __dependency4__, __exports__) {
    
    var parser = __dependency1__["default"];
    var AST = __dependency2__["default"];
    var Helpers = __dependency3__;
    var extend = __dependency4__.extend;

    __exports__.parser = parser;

    var yy = {};
    extend(yy, Helpers, AST);

    function parse(input) {
      // Just return if an already-compile AST was passed in.
      if (input.constructor === AST.ProgramNode) { return input; }

      parser.yy = yy;

      return parser.parse(input);
    }

    __exports__.parse = parse;
  });
define(
  'handlebars/compiler/compiler',["../exception","../utils","exports"],
  function(__dependency1__, __dependency2__, __exports__) {
    
    var Exception = __dependency1__["default"];
    var isArray = __dependency2__.isArray;

    var slice = [].slice;

    function Compiler() {}

    __exports__.Compiler = Compiler;// the foundHelper register will disambiguate helper lookup from finding a
    // function in a context. This is necessary for mustache compatibility, which
    // requires that context functions in blocks are evaluated by blockHelperMissing,
    // and then proceed as if the resulting value was provided to blockHelperMissing.

    Compiler.prototype = {
      compiler: Compiler,

      equals: function(other) {
        var len = this.opcodes.length;
        if (other.opcodes.length !== len) {
          return false;
        }

        for (var i = 0; i < len; i++) {
          var opcode = this.opcodes[i],
              otherOpcode = other.opcodes[i];
          if (opcode.opcode !== otherOpcode.opcode || !argEquals(opcode.args, otherOpcode.args)) {
            return false;
          }
        }

        // We know that length is the same between the two arrays because they are directly tied
        // to the opcode behavior above.
        len = this.children.length;
        for (i = 0; i < len; i++) {
          if (!this.children[i].equals(other.children[i])) {
            return false;
          }
        }

        return true;
      },

      guid: 0,

      compile: function(program, options) {
        this.opcodes = [];
        this.children = [];
        this.depths = {list: []};
        this.options = options;
        this.stringParams = options.stringParams;
        this.trackIds = options.trackIds;

        // These changes will propagate to the other compiler components
        var knownHelpers = this.options.knownHelpers;
        this.options.knownHelpers = {
          'helperMissing': true,
          'blockHelperMissing': true,
          'each': true,
          'if': true,
          'unless': true,
          'with': true,
          'log': true,
          'lookup': true
        };
        if (knownHelpers) {
          for (var name in knownHelpers) {
            this.options.knownHelpers[name] = knownHelpers[name];
          }
        }

        return this.accept(program);
      },

      accept: function(node) {
        return this[node.type](node);
      },

      program: function(program) {
        var statements = program.statements;

        for(var i=0, l=statements.length; i<l; i++) {
          this.accept(statements[i]);
        }
        this.isSimple = l === 1;

        this.depths.list = this.depths.list.sort(function(a, b) {
          return a - b;
        });

        return this;
      },

      compileProgram: function(program) {
        var result = new this.compiler().compile(program, this.options);
        var guid = this.guid++, depth;

        this.usePartial = this.usePartial || result.usePartial;

        this.children[guid] = result;

        for(var i=0, l=result.depths.list.length; i<l; i++) {
          depth = result.depths.list[i];

          if(depth < 2) { continue; }
          else { this.addDepth(depth - 1); }
        }

        return guid;
      },

      block: function(block) {
        var mustache = block.mustache,
            program = block.program,
            inverse = block.inverse;

        if (program) {
          program = this.compileProgram(program);
        }

        if (inverse) {
          inverse = this.compileProgram(inverse);
        }

        var sexpr = mustache.sexpr;
        var type = this.classifySexpr(sexpr);

        if (type === "helper") {
          this.helperSexpr(sexpr, program, inverse);
        } else if (type === "simple") {
          this.simpleSexpr(sexpr);

          // now that the simple mustache is resolved, we need to
          // evaluate it by executing `blockHelperMissing`
          this.opcode('pushProgram', program);
          this.opcode('pushProgram', inverse);
          this.opcode('emptyHash');
          this.opcode('blockValue', sexpr.id.original);
        } else {
          this.ambiguousSexpr(sexpr, program, inverse);

          // now that the simple mustache is resolved, we need to
          // evaluate it by executing `blockHelperMissing`
          this.opcode('pushProgram', program);
          this.opcode('pushProgram', inverse);
          this.opcode('emptyHash');
          this.opcode('ambiguousBlockValue');
        }

        this.opcode('append');
      },

      hash: function(hash) {
        var pairs = hash.pairs, i, l;

        this.opcode('pushHash');

        for(i=0, l=pairs.length; i<l; i++) {
          this.pushParam(pairs[i][1]);
        }
        while(i--) {
          this.opcode('assignToHash', pairs[i][0]);
        }
        this.opcode('popHash');
      },

      partial: function(partial) {
        var partialName = partial.partialName;
        this.usePartial = true;

        if (partial.hash) {
          this.accept(partial.hash);
        } else {
          this.opcode('push', 'undefined');
        }

        if (partial.context) {
          this.accept(partial.context);
        } else {
          this.opcode('getContext', 0);
          this.opcode('pushContext');
        }

        this.opcode('invokePartial', partialName.name, partial.indent || '');
        this.opcode('append');
      },

      content: function(content) {
        if (content.string) {
          this.opcode('appendContent', content.string);
        }
      },

      mustache: function(mustache) {
        this.sexpr(mustache.sexpr);

        if(mustache.escaped && !this.options.noEscape) {
          this.opcode('appendEscaped');
        } else {
          this.opcode('append');
        }
      },

      ambiguousSexpr: function(sexpr, program, inverse) {
        var id = sexpr.id,
            name = id.parts[0],
            isBlock = program != null || inverse != null;

        this.opcode('getContext', id.depth);

        this.opcode('pushProgram', program);
        this.opcode('pushProgram', inverse);

        this.ID(id);

        this.opcode('invokeAmbiguous', name, isBlock);
      },

      simpleSexpr: function(sexpr) {
        var id = sexpr.id;

        if (id.type === 'DATA') {
          this.DATA(id);
        } else if (id.parts.length) {
          this.ID(id);
        } else {
          // Simplified ID for `this`
          this.addDepth(id.depth);
          this.opcode('getContext', id.depth);
          this.opcode('pushContext');
        }

        this.opcode('resolvePossibleLambda');
      },

      helperSexpr: function(sexpr, program, inverse) {
        var params = this.setupFullMustacheParams(sexpr, program, inverse),
            id = sexpr.id,
            name = id.parts[0];

        if (this.options.knownHelpers[name]) {
          this.opcode('invokeKnownHelper', params.length, name);
        } else if (this.options.knownHelpersOnly) {
          throw new Exception("You specified knownHelpersOnly, but used the unknown helper " + name, sexpr);
        } else {
          id.falsy = true;

          this.ID(id);
          this.opcode('invokeHelper', params.length, id.original, id.isSimple);
        }
      },

      sexpr: function(sexpr) {
        var type = this.classifySexpr(sexpr);

        if (type === "simple") {
          this.simpleSexpr(sexpr);
        } else if (type === "helper") {
          this.helperSexpr(sexpr);
        } else {
          this.ambiguousSexpr(sexpr);
        }
      },

      ID: function(id) {
        this.addDepth(id.depth);
        this.opcode('getContext', id.depth);

        var name = id.parts[0];
        if (!name) {
          // Context reference, i.e. `{{foo .}}` or `{{foo ..}}`
          this.opcode('pushContext');
        } else {
          this.opcode('lookupOnContext', id.parts, id.falsy, id.isScoped);
        }
      },

      DATA: function(data) {
        this.options.data = true;
        this.opcode('lookupData', data.id.depth, data.id.parts);
      },

      STRING: function(string) {
        this.opcode('pushString', string.string);
      },

      NUMBER: function(number) {
        this.opcode('pushLiteral', number.number);
      },

      BOOLEAN: function(bool) {
        this.opcode('pushLiteral', bool.bool);
      },

      comment: function() {},

      // HELPERS
      opcode: function(name) {
        this.opcodes.push({ opcode: name, args: slice.call(arguments, 1) });
      },

      addDepth: function(depth) {
        if(depth === 0) { return; }

        if(!this.depths[depth]) {
          this.depths[depth] = true;
          this.depths.list.push(depth);
        }
      },

      classifySexpr: function(sexpr) {
        var isHelper   = sexpr.isHelper;
        var isEligible = sexpr.eligibleHelper;
        var options    = this.options;

        // if ambiguous, we can possibly resolve the ambiguity now
        // An eligible helper is one that does not have a complex path, i.e. `this.foo`, `../foo` etc.
        if (isEligible && !isHelper) {
          var name = sexpr.id.parts[0];

          if (options.knownHelpers[name]) {
            isHelper = true;
          } else if (options.knownHelpersOnly) {
            isEligible = false;
          }
        }

        if (isHelper) { return "helper"; }
        else if (isEligible) { return "ambiguous"; }
        else { return "simple"; }
      },

      pushParams: function(params) {
        for(var i=0, l=params.length; i<l; i++) {
          this.pushParam(params[i]);
        }
      },

      pushParam: function(val) {
        if (this.stringParams) {
          if(val.depth) {
            this.addDepth(val.depth);
          }
          this.opcode('getContext', val.depth || 0);
          this.opcode('pushStringParam', val.stringModeValue, val.type);

          if (val.type === 'sexpr') {
            // Subexpressions get evaluated and passed in
            // in string params mode.
            this.sexpr(val);
          }
        } else {
          if (this.trackIds) {
            this.opcode('pushId', val.type, val.idName || val.stringModeValue);
          }
          this.accept(val);
        }
      },

      setupFullMustacheParams: function(sexpr, program, inverse) {
        var params = sexpr.params;
        this.pushParams(params);

        this.opcode('pushProgram', program);
        this.opcode('pushProgram', inverse);

        if (sexpr.hash) {
          this.hash(sexpr.hash);
        } else {
          this.opcode('emptyHash');
        }

        return params;
      }
    };

    function precompile(input, options, env) {
      if (input == null || (typeof input !== 'string' && input.constructor !== env.AST.ProgramNode)) {
        throw new Exception("You must pass a string or Handlebars AST to Handlebars.precompile. You passed " + input);
      }

      options = options || {};
      if (!('data' in options)) {
        options.data = true;
      }
      if (options.compat) {
        options.useDepths = true;
      }

      var ast = env.parse(input);
      var environment = new env.Compiler().compile(ast, options);
      return new env.JavaScriptCompiler().compile(environment, options);
    }

    __exports__.precompile = precompile;function compile(input, options, env) {
      if (input == null || (typeof input !== 'string' && input.constructor !== env.AST.ProgramNode)) {
        throw new Exception("You must pass a string or Handlebars AST to Handlebars.compile. You passed " + input);
      }

      options = options || {};

      if (!('data' in options)) {
        options.data = true;
      }
      if (options.compat) {
        options.useDepths = true;
      }

      var compiled;

      function compileInput() {
        var ast = env.parse(input);
        var environment = new env.Compiler().compile(ast, options);
        var templateSpec = new env.JavaScriptCompiler().compile(environment, options, undefined, true);
        return env.template(templateSpec);
      }

      // Template is only compiled on first use and cached after that point.
      var ret = function(context, options) {
        if (!compiled) {
          compiled = compileInput();
        }
        return compiled.call(this, context, options);
      };
      ret._setup = function(options) {
        if (!compiled) {
          compiled = compileInput();
        }
        return compiled._setup(options);
      };
      ret._child = function(i, data, depths) {
        if (!compiled) {
          compiled = compileInput();
        }
        return compiled._child(i, data, depths);
      };
      return ret;
    }

    __exports__.compile = compile;function argEquals(a, b) {
      if (a === b) {
        return true;
      }

      if (isArray(a) && isArray(b) && a.length === b.length) {
        for (var i = 0; i < a.length; i++) {
          if (!argEquals(a[i], b[i])) {
            return false;
          }
        }
        return true;
      }
    }
  });
define(
  'handlebars/compiler/javascript-compiler',["../base","../exception","exports"],
  function(__dependency1__, __dependency2__, __exports__) {
    
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
define(
  'handlebars',["./handlebars.runtime","./handlebars/compiler/ast","./handlebars/compiler/base","./handlebars/compiler/compiler","./handlebars/compiler/javascript-compiler","exports"],
  function(__dependency1__, __dependency2__, __dependency3__, __dependency4__, __dependency5__, __exports__) {
    
    /*globals Handlebars: true */
    var Handlebars = __dependency1__["default"];

    // Compiler imports
    var AST = __dependency2__["default"];
    var Parser = __dependency3__.parser;
    var parse = __dependency3__.parse;
    var Compiler = __dependency4__.Compiler;
    var compile = __dependency4__.compile;
    var precompile = __dependency4__.precompile;
    var JavaScriptCompiler = __dependency5__["default"];

    var _create = Handlebars.create;
    var create = function() {
      var hb = _create();

      hb.compile = function(input, options) {
        return compile(input, options, hb);
      };
      hb.precompile = function (input, options) {
        return precompile(input, options, hb);
      };

      hb.AST = AST;
      hb.Compiler = Compiler;
      hb.JavaScriptCompiler = JavaScriptCompiler;
      hb.Parser = Parser;
      hb.parse = parse;

      return hb;
    };

    Handlebars = create();
    Handlebars.create = create;

    Handlebars['default'] = Handlebars;

    __exports__["default"] = Handlebars;
  });