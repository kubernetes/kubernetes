// swagger.js
// version 2.0.47

(function () {

  var __bind = function (fn, me) {
    return function () {
      return fn.apply(me, arguments);
    };
  };

  var log = function () {
    log.history = log.history || [];
    log.history.push(arguments);
    if (this.console) {
      console.log(Array.prototype.slice.call(arguments)[0]);
    }
  };

  // if you want to apply conditional formatting of parameter values
  var parameterMacro = function (value) {
    return value;
  }

  // if you want to apply conditional formatting of model property values
  var modelPropertyMacro = function (value) {
    return value;
  }

  if (!Array.prototype.indexOf) {
    Array.prototype.indexOf = function (obj, start) {
      for (var i = (start || 0), j = this.length; i < j; i++) {
        if (this[i] === obj) { return i; }
      }
      return -1;
    }
  }

  if (!('filter' in Array.prototype)) {
    Array.prototype.filter = function (filter, that /*opt*/) {
      var other = [], v;
      for (var i = 0, n = this.length; i < n; i++)
        if (i in this && filter.call(that, v = this[i], i, this))
          other.push(v);
      return other;
    };
  }

  if (!('map' in Array.prototype)) {
    Array.prototype.map = function (mapper, that /*opt*/) {
      var other = new Array(this.length);
      for (var i = 0, n = this.length; i < n; i++)
        if (i in this)
          other[i] = mapper.call(that, this[i], i, this);
      return other;
    };
  }

  Object.keys = Object.keys || (function () {
    var hasOwnProperty = Object.prototype.hasOwnProperty,
      hasDontEnumBug = !{ toString: null }.propertyIsEnumerable("toString"),
      DontEnums = [
      'toString',
      'toLocaleString',
      'valueOf',
      'hasOwnProperty',
      'isPrototypeOf',
      'propertyIsEnumerable',
      'constructor'
      ],
    DontEnumsLength = DontEnums.length;

    return function (o) {
      if (typeof o != "object" && typeof o != "function" || o === null)
        throw new TypeError("Object.keys called on a non-object");

      var result = [];
      for (var name in o) {
        if (hasOwnProperty.call(o, name))
          result.push(name);
      }

      if (hasDontEnumBug) {
        for (var i = 0; i < DontEnumsLength; i++) {
          if (hasOwnProperty.call(o, DontEnums[i]))
            result.push(DontEnums[i]);
        }
      }

      return result;
    };
  })();

  var SwaggerApi = function (url, options) {
    this.isBuilt = false;
    this.url = null;
    this.debug = false;
    this.basePath = null;
    this.authorizations = null;
    this.authorizationScheme = null;
    this.info = null;
    this.useJQuery = false;
    this.modelsArray = [];
    this.isValid;

    options = (options || {});
    if (url)
      if (url.url)
        options = url;
      else
        this.url = url;
    else
      options = url;

    if (options.url != null)
      this.url = options.url;

    if (options.success != null)
      this.success = options.success;

    if (typeof options.useJQuery === 'boolean')
      this.useJQuery = options.useJQuery;

    this.failure = options.failure != null ? options.failure : function () { };
    this.progress = options.progress != null ? options.progress : function () { };
    if (options.success != null) {
      this.build();
      this.isBuilt = true;
    }
  }

  SwaggerApi.prototype.build = function () {
    if (this.isBuilt)
      return this;
    var _this = this;
    this.progress('fetching resource list: ' + this.url);
    var obj = {
      useJQuery: this.useJQuery,
      url: this.url,
      method: "get",
      headers: {
        accept: "application/json,application/json;charset=utf-8,*/*"
      },
      on: {
        error: function (response) {
          if (_this.url.substring(0, 4) !== 'http') {
            return _this.fail('Please specify the protocol for ' + _this.url);
          } else if (response.status === 0) {
            return _this.fail('Can\'t read from server.  It may not have the appropriate access-control-origin settings.');
          } else if (response.status === 404) {
            return _this.fail('Can\'t read swagger JSON from ' + _this.url);
          } else {
            return _this.fail(response.status + ' : ' + response.statusText + ' ' + _this.url);
          }
        },
        response: function (resp) {
          var responseObj = resp.obj || JSON.parse(resp.data);
          _this.swaggerVersion = responseObj.swaggerVersion;
          if (_this.swaggerVersion === "1.2") {
            return _this.buildFromSpec(responseObj);
          } else {
            return _this.buildFrom1_1Spec(responseObj);
          }
        }
      }
    };
    var e = (typeof window !== 'undefined' ? window : exports);
    e.authorizations.apply(obj);
    new SwaggerHttp().execute(obj);
    return this;
  };

  SwaggerApi.prototype.buildFromSpec = function (response) {
    if (response.apiVersion != null) {
      this.apiVersion = response.apiVersion;
    }
    this.apis = {};
    this.apisArray = [];
    this.consumes = response.consumes;
    this.produces = response.produces;
    this.authSchemes = response.authorizations;
    if (response.info != null) {
      this.info = response.info;
    }
    var isApi = false;
    var i;
    for (i = 0; i < response.apis.length; i++) {
      var api = response.apis[i];
      if (api.operations) {
        var j;
        for (j = 0; j < api.operations.length; j++) {
          operation = api.operations[j];
          isApi = true;
        }
      }
    }
    if (response.basePath)
      this.basePath = response.basePath;
    else if (this.url.indexOf('?') > 0)
      this.basePath = this.url.substring(0, this.url.lastIndexOf('?'));
    else
      this.basePath = this.url;

    if (isApi) {
      var newName = response.resourcePath.replace(/\//g, '');
      this.resourcePath = response.resourcePath;
      var res = new SwaggerResource(response, this);
      this.apis[newName] = res;
      this.apisArray.push(res);
    } else {
      var k;
      for (k = 0; k < response.apis.length; k++) {
        var resource = response.apis[k];
        var res = new SwaggerResource(resource, this);
        this.apis[res.name] = res;
        this.apisArray.push(res);
      }
    }
    this.isValid = true;
    if (this.success) {
      this.success();
    }
    return this;
  };

  SwaggerApi.prototype.buildFrom1_1Spec = function (response) {
    log("This API is using a deprecated version of Swagger!  Please see http://github.com/wordnik/swagger-core/wiki for more info");
    if (response.apiVersion != null)
      this.apiVersion = response.apiVersion;
    this.apis = {};
    this.apisArray = [];
    this.produces = response.produces;
    if (response.info != null) {
      this.info = response.info;
    }
    var isApi = false;
    for (var i = 0; i < response.apis.length; i++) {
      var api = response.apis[i];
      if (api.operations) {
        for (var j = 0; j < api.operations.length; j++) {
          operation = api.operations[j];
          isApi = true;
        }
      }
    }
    if (response.basePath) {
      this.basePath = response.basePath;
    } else if (this.url.indexOf('?') > 0) {
      this.basePath = this.url.substring(0, this.url.lastIndexOf('?'));
    } else {
      this.basePath = this.url;
    }
    if (isApi) {
      var newName = response.resourcePath.replace(/\//g, '');
      this.resourcePath = response.resourcePath;
      var res = new SwaggerResource(response, this);
      this.apis[newName] = res;
      this.apisArray.push(res);
    } else {
      for (k = 0; k < response.apis.length; k++) {
        resource = response.apis[k];
        var res = new SwaggerResource(resource, this);
        this.apis[res.name] = res;
        this.apisArray.push(res);
      }
    }
    this.isValid = true;
    if (this.success) {
      this.success();
    }
    return this;
  };

  SwaggerApi.prototype.selfReflect = function () {
    var resource, resource_name, _ref;
    if (this.apis == null) {
      return false;
    }
    _ref = this.apis;
    for (resource_name in _ref) {
      resource = _ref[resource_name];
      if (resource.ready == null) {
        return false;
      }
    }
    this.setConsolidatedModels();
    this.ready = true;
    if (this.success != null) {
      return this.success();
    }
  };

  SwaggerApi.prototype.fail = function (message) {
    this.failure(message);
    throw message;
  };

  SwaggerApi.prototype.setConsolidatedModels = function () {
    var model, modelName, resource, resource_name, _i, _len, _ref, _ref1, _results;
    this.models = {};
    _ref = this.apis;
    for (resource_name in _ref) {
      resource = _ref[resource_name];
      for (modelName in resource.models) {
        if (this.models[modelName] == null) {
          this.models[modelName] = resource.models[modelName];
          this.modelsArray.push(resource.models[modelName]);
        }
      }
    }
    _ref1 = this.modelsArray;
    _results = [];
    for (_i = 0, _len = _ref1.length; _i < _len; _i++) {
      model = _ref1[_i];
      _results.push(model.setReferencedModels(this.models));
    }
    return _results;
  };

  SwaggerApi.prototype.help = function () {
    var operation, operation_name, parameter, resource, resource_name, _i, _len, _ref, _ref1, _ref2;
    _ref = this.apis;
    for (resource_name in _ref) {
      resource = _ref[resource_name];
      log(resource_name);
      _ref1 = resource.operations;
      for (operation_name in _ref1) {
        operation = _ref1[operation_name];
        log("  " + operation.nickname);
        _ref2 = operation.parameters;
        for (_i = 0, _len = _ref2.length; _i < _len; _i++) {
          parameter = _ref2[_i];
          log("  " + parameter.name + (parameter.required ? ' (required)' : '') + " - " + parameter.description);
        }
      }
    }
    return this;
  };

  var SwaggerResource = function (resourceObj, api) {
    var _this = this;
    this.api = api;
    this.api = this.api;
    var consumes = (this.consumes | []);
    var produces = (this.produces | []);
    this.path = this.api.resourcePath != null ? this.api.resourcePath : resourceObj.path;
    this.description = resourceObj.description;

    var parts = this.path.split("/");
    this.name = parts[parts.length - 1].replace('.{format}', '');
    this.basePath = this.api.basePath;
    this.operations = {};
    this.operationsArray = [];
    this.modelsArray = [];
    this.models = {};
    this.rawModels = {};
    this.useJQuery = (typeof api.useJQuery !== 'undefined' ? api.useJQuery : null);

    if ((resourceObj.apis != null) && (this.api.resourcePath != null)) {
      this.addApiDeclaration(resourceObj);
    } else {
      if (this.path == null) {
        this.api.fail("SwaggerResources must have a path.");
      }
      if (this.path.substring(0, 4) === 'http') {
        this.url = this.path.replace('{format}', 'json');
      } else {
        this.url = this.api.basePath + this.path.replace('{format}', 'json');
      }
      this.api.progress('fetching resource ' + this.name + ': ' + this.url);
      var obj = {
        url: this.url,
        method: "get",
        useJQuery: this.useJQuery,
        headers: {
          accept: "application/json,application/json;charset=utf-8,*/*"
        },
        on: {
          response: function (resp) {
            var responseObj = resp.obj || JSON.parse(resp.data);
            return _this.addApiDeclaration(responseObj);
          },
          error: function (response) {
            return _this.api.fail("Unable to read api '" +
              _this.name + "' from path " + _this.url + " (server returned " + response.statusText + ")");
          }
        }
      };
      var e = typeof window !== 'undefined' ? window : exports;
      e.authorizations.apply(obj);
      new SwaggerHttp().execute(obj);
    }
  }

  SwaggerResource.prototype.getAbsoluteBasePath = function (relativeBasePath) {
    var pos, url;
    url = this.api.basePath;
    pos = url.lastIndexOf(relativeBasePath);
    var parts = url.split("/");
    var rootUrl = parts[0] + "//" + parts[2];

    if (relativeBasePath.indexOf("http") === 0)
      return relativeBasePath;
    if (relativeBasePath === "/")
      return rootUrl;
    if (relativeBasePath.substring(0, 1) == "/") {
      // use root + relative
      return rootUrl + relativeBasePath;
    }
    else {
      var pos = this.basePath.lastIndexOf("/");
      var base = this.basePath.substring(0, pos);
      if (base.substring(base.length - 1) == "/")
        return base + relativeBasePath;
      else
        return base + "/" + relativeBasePath;
    }
  };

  SwaggerResource.prototype.addApiDeclaration = function (response) {
    if (response.produces != null)
      this.produces = response.produces;
    if (response.consumes != null)
      this.consumes = response.consumes;
    if ((response.basePath != null) && response.basePath.replace(/\s/g, '').length > 0)
      this.basePath = response.basePath.indexOf("http") === -1 ? this.getAbsoluteBasePath(response.basePath) : response.basePath;

    this.addModels(response.models);
    if (response.apis) {
      for (var i = 0 ; i < response.apis.length; i++) {
        var endpoint = response.apis[i];
        this.addOperations(endpoint.path, endpoint.operations, response.consumes, response.produces);
      }
    }
    this.api[this.name] = this;
    this.ready = true;
    return this.api.selfReflect();
  };

  SwaggerResource.prototype.addModels = function (models) {
    if (models != null) {
      var modelName;
      for (modelName in models) {
        if (this.models[modelName] == null) {
          var swaggerModel = new SwaggerModel(modelName, models[modelName]);
          this.modelsArray.push(swaggerModel);
          this.models[modelName] = swaggerModel;
          this.rawModels[modelName] = models[modelName];
        }
      }
      var output = [];
      for (var i = 0; i < this.modelsArray.length; i++) {
        var model = this.modelsArray[i];
        output.push(model.setReferencedModels(this.models));
      }
      return output;
    }
  };

  SwaggerResource.prototype.addOperations = function (resource_path, ops, consumes, produces) {
    if (ops) {
      var output = [];
      for (var i = 0; i < ops.length; i++) {
        var o = ops[i];
        consumes = this.consumes;
        produces = this.produces;
        if (o.consumes != null)
          consumes = o.consumes;
        else
          consumes = this.consumes;

        if (o.produces != null)
          produces = o.produces;
        else
          produces = this.produces;
        var type = (o.type || o.responseClass);

        if (type === "array") {
          ref = null;
          if (o.items)
            ref = o.items["type"] || o.items["$ref"];
          type = "array[" + ref + "]";
        }
        var responseMessages = o.responseMessages;
        var method = o.method;
        if (o.httpMethod) {
          method = o.httpMethod;
        }
        if (o.supportedContentTypes) {
          consumes = o.supportedContentTypes;
        }
        if (o.errorResponses) {
          responseMessages = o.errorResponses;
          for (var j = 0; j < responseMessages.length; j++) {
            r = responseMessages[j];
            r.message = r.reason;
            r.reason = null;
          }
        }
        o.nickname = this.sanitize(o.nickname);
        var op = new SwaggerOperation(o.nickname, resource_path, method, o.parameters, o.summary, o.notes, type, responseMessages, this, consumes, produces, o.authorizations, o.deprecated);
        this.operations[op.nickname] = op;
        output.push(this.operationsArray.push(op));
      }
      return output;
    }
  };

  SwaggerResource.prototype.sanitize = function (nickname) {
    var op;
    op = nickname.replace(/[\s!@#$%^&*()_+=\[{\]};:<>|.\/?,\\'""-]/g, '_');
    op = op.replace(/((_){2,})/g, '_');
    op = op.replace(/^(_)*/g, '');
    op = op.replace(/([_])*$/g, '');
    return op;
  };

  SwaggerResource.prototype.help = function () {
    var op = this.operations;
    var output = [];
    var operation_name;
    for (operation_name in op) {
      operation = op[operation_name];
      var msg = "  " + operation.nickname;
      for (var i = 0; i < operation.parameters; i++) {
        parameter = operation.parameters[i];
        msg.concat("  " + parameter.name + (parameter.required ? ' (required)' : '') + " - " + parameter.description);
      }
      output.push(msg);
    }
    return output;
  };

  var SwaggerModel = function (modelName, obj) {
    this.name = obj.id != null ? obj.id : modelName;
    this.properties = [];
    var propertyName;
    for (propertyName in obj.properties) {
      if (obj.required != null) {
        var value;
        for (value in obj.required) {
          if (propertyName === obj.required[value]) {
            obj.properties[propertyName].required = true;
          }
        }
      }
      var prop = new SwaggerModelProperty(propertyName, obj.properties[propertyName]);
      this.properties.push(prop);
    }
  }

  SwaggerModel.prototype.setReferencedModels = function (allModels) {
    var results = [];
    for (var i = 0; i < this.properties.length; i++) {
      var property = this.properties[i];
      var type = property.type || property.dataType;
      if (allModels[type] != null)
        results.push(property.refModel = allModels[type]);
      else if ((property.refDataType != null) && (allModels[property.refDataType] != null))
        results.push(property.refModel = allModels[property.refDataType]);
      else
        results.push(void 0);
    }
    return results;
  };

  SwaggerModel.prototype.getMockSignature = function (modelsToIgnore) {
    var propertiesStr = [];
    for (var i = 0; i < this.properties.length; i++) {
      var prop = this.properties[i];
      propertiesStr.push(prop.toString());
    }

    var strong = '<span class="strong">';
    var stronger = '<span class="stronger">';
    var strongClose = '</span>';
    var classOpen = strong + this.name + ' {' + strongClose;
    var classClose = strong + '}' + strongClose;
    var returnVal = classOpen + '<div>' + propertiesStr.join(',</div><div>') + '</div>' + classClose;
    if (!modelsToIgnore)
      modelsToIgnore = [];
    modelsToIgnore.push(this.name);

    for (var i = 0; i < this.properties.length; i++) {
      var prop = this.properties[i];
      if ((prop.refModel != null) && modelsToIgnore.indexOf(prop.refModel.name) === -1) {
        returnVal = returnVal + ('<br>' + prop.refModel.getMockSignature(modelsToIgnore));
      }
    }
    return returnVal;
  };

  SwaggerModel.prototype.createJSONSample = function (modelsToIgnore) {
    if (sampleModels[this.name]) {
      return sampleModels[this.name];
    }
    else {
      var result = {};
      var modelsToIgnore = (modelsToIgnore || [])
      modelsToIgnore.push(this.name);
      for (var i = 0; i < this.properties.length; i++) {
        var prop = this.properties[i];
        result[prop.name] = prop.getSampleValue(modelsToIgnore);
      }
      modelsToIgnore.pop(this.name);
      return result;
    }
  };

  var SwaggerModelProperty = function (name, obj) {
    this.name = name;
    this.dataType = obj.type || obj.dataType || obj["$ref"];
    this.isCollection = this.dataType && (this.dataType.toLowerCase() === 'array' || this.dataType.toLowerCase() === 'list' || this.dataType.toLowerCase() === 'set');
    this.descr = obj.description;
    this.required = obj.required;
    this.defaultValue = modelPropertyMacro(obj.defaultValue);
    if (obj.items != null) {
      if (obj.items.type != null) {
        this.refDataType = obj.items.type;
      }
      if (obj.items.$ref != null) {
        this.refDataType = obj.items.$ref;
      }
    }
    this.dataTypeWithRef = this.refDataType != null ? (this.dataType + '[' + this.refDataType + ']') : this.dataType;
    if (obj.allowableValues != null) {
      this.valueType = obj.allowableValues.valueType;
      this.values = obj.allowableValues.values;
      if (this.values != null) {
        this.valuesString = "'" + this.values.join("' or '") + "'";
      }
    }
    if (obj["enum"] != null) {
      this.valueType = "string";
      this.values = obj["enum"];
      if (this.values != null) {
        this.valueString = "'" + this.values.join("' or '") + "'";
      }
    }
  }

  SwaggerModelProperty.prototype.getSampleValue = function (modelsToIgnore) {
    var result;
    if ((this.refModel != null) && (modelsToIgnore.indexOf(this.refModel.name) === -1)) {
      result = this.refModel.createJSONSample(modelsToIgnore);
    } else {
      if (this.isCollection) {
        result = this.toSampleValue(this.refDataType);
      } else {
        result = this.toSampleValue(this.dataType);
      }
    }
    if (this.isCollection) {
      return [result];
    } else {
      return result;
    }
  };

  SwaggerModelProperty.prototype.toSampleValue = function (value) {
    var result;
    if ((typeof this.defaultValue !== 'undefined') && this.defaultValue !== null) {
      result = this.defaultValue;
    } else if (value === "integer") {
      result = 0;
    } else if (value === "boolean") {
      result = false;
    } else if (value === "double" || value === "number") {
      result = 0.0;
    } else if (value === "string") {
      result = "";
    } else {
      result = value;
    }
    return result;
  };

  SwaggerModelProperty.prototype.toString = function () {
    var req = this.required ? 'propReq' : 'propOpt';
    var str = '<span class="propName ' + req + '">' + this.name + '</span> (<span class="propType">' + this.dataTypeWithRef + '</span>';
    if (!this.required) {
      str += ', <span class="propOptKey">optional</span>';
    }
    str += ')';
    if (this.values != null) {
      str += " = <span class='propVals'>['" + this.values.join("' or '") + "']</span>";
    }
    if (this.descr != null) {
      str += ': <span class="propDesc">' + this.descr + '</span>';
    }
    return str;
  };

  var SwaggerOperation = function (nickname, path, method, parameters, summary, notes, type, responseMessages, resource, consumes, produces, authorizations, deprecated) {
    var _this = this;

    var errors = [];
    this.nickname = (nickname || errors.push("SwaggerOperations must have a nickname."));
    this.path = (path || errors.push("SwaggerOperation " + nickname + " is missing path."));
    this.method = (method || errors.push("SwaggerOperation " + nickname + " is missing method."));
    this.parameters = parameters != null ? parameters : [];
    this.summary = summary;
    this.notes = notes;
    this.type = type;
    this.responseMessages = (responseMessages || []);
    this.resource = (resource || errors.push("Resource is required"));
    this.consumes = consumes;
    this.produces = produces;
    this.authorizations = authorizations;
    this.deprecated = deprecated;
    this["do"] = __bind(this["do"], this);

    if (errors.length > 0) {
      console.error('SwaggerOperation errors', errors, arguments);
      this.resource.api.fail(errors);
    }

    this.path = this.path.replace('{format}', 'json');
    this.method = this.method.toLowerCase();
    this.isGetMethod = this.method === "get";

    this.resourceName = this.resource.name;
    if (typeof this.type !== 'undefined' && this.type === 'void')
      this.type = null;
    else {
      this.responseClassSignature = this.getSignature(this.type, this.resource.models);
      this.responseSampleJSON = this.getSampleJSON(this.type, this.resource.models);
    }

    for (var i = 0; i < this.parameters.length; i++) {
      var param = this.parameters[i];
      // might take this away
      param.name = param.name || param.type || param.dataType;

      // for 1.1 compatibility
      var type = param.type || param.dataType;
      if (type === 'array') {
        type = 'array[' + (param.items.$ref ? param.items.$ref : param.items.type) + ']';
      }
      param.type = type;

      if (type && type.toLowerCase() === 'boolean') {
        param.allowableValues = {};
        param.allowableValues.values = ["true", "false"];
      }
      param.signature = this.getSignature(type, this.resource.models);
      param.sampleJSON = this.getSampleJSON(type, this.resource.models);

      var enumValue = param["enum"];
      if (enumValue != null) {
        param.isList = true;
        param.allowableValues = {};
        param.allowableValues.descriptiveValues = [];

        for (var j = 0; j < enumValue.length; j++) {
          var v = enumValue[j];
          if (param.defaultValue != null) {
            param.allowableValues.descriptiveValues.push({
              value: String(v),
              isDefault: (v === param.defaultValue)
            });
          }
          else {
            param.allowableValues.descriptiveValues.push({
              value: String(v),
              isDefault: false
            });
          }
        }
      }
      else if (param.allowableValues != null) {
        if (param.allowableValues.valueType === "RANGE")
          param.isRange = true;
        else
          param.isList = true;
        if (param.allowableValues != null) {
          param.allowableValues.descriptiveValues = [];
          if (param.allowableValues.values) {
            for (var j = 0; j < param.allowableValues.values.length; j++) {
              var v = param.allowableValues.values[j];
              if (param.defaultValue != null) {
                param.allowableValues.descriptiveValues.push({
                  value: String(v),
                  isDefault: (v === param.defaultValue)
                });
              }
              else {
                param.allowableValues.descriptiveValues.push({
                  value: String(v),
                  isDefault: false
                });
              }
            }
          }
        }
      }
      param.defaultValue = parameterMacro(param.defaultValue);
    }
    this.resource[this.nickname] = function (args, callback, error) {
      return _this["do"](args, callback, error);
    };
    this.resource[this.nickname].help = function () {
      return _this.help();
    };
  }

  SwaggerOperation.prototype.isListType = function (type) {
    if (type && type.indexOf('[') >= 0) {
      return type.substring(type.indexOf('[') + 1, type.indexOf(']'));
    } else {
      return void 0;
    }
  };

  SwaggerOperation.prototype.getSignature = function (type, models) {
    var isPrimitive, listType;
    listType = this.isListType(type);
    isPrimitive = ((listType != null) && models[listType]) || (models[type] != null) ? false : true;
    if (isPrimitive) {
      return type;
    } else {
      if (listType != null) {
        return models[listType].getMockSignature();
      } else {
        return models[type].getMockSignature();
      }
    }
  };

  SwaggerOperation.prototype.getSampleJSON = function (type, models) {
    var isPrimitive, listType, val;
    listType = this.isListType(type);
    isPrimitive = ((listType != null) && models[listType]) || (models[type] != null) ? false : true;
    val = isPrimitive ? void 0 : (listType != null ? models[listType].createJSONSample() : models[type].createJSONSample());
    if (val) {
      val = listType ? [val] : val;
      if (typeof val == "string")
        return val;
      else if (typeof val === "object") {
        var t = val;
        if (val instanceof Array && val.length > 0) {
          t = val[0];
        }
        if (t.nodeName) {
          var xmlString = new XMLSerializer().serializeToString(t);
          return this.formatXml(xmlString);
        }
        else
          return JSON.stringify(val, null, 2);
      }
      else
        return val;
    }
  };

  SwaggerOperation.prototype["do"] = function (args, opts, callback, error) {
    var key, param, params, possibleParams, req, requestContentType, responseContentType, value, _i, _len, _ref;
    if (args == null) {
      args = {};
    }
    if (opts == null) {
      opts = {};
    }
    requestContentType = null;
    responseContentType = null;
    if ((typeof args) === "function") {
      error = opts;
      callback = args;
      args = {};
    }
    if ((typeof opts) === "function") {
      error = callback;
      callback = opts;
    }
    if (error == null) {
      error = function (xhr, textStatus, error) {
        return log(xhr, textStatus, error);
      };
    }
    if (callback == null) {
      callback = function (response) {
        var content;
        content = null;
        if (response != null) {
          content = response.data;
        } else {
          content = "no data";
        }
        return log("default callback: " + content);
      };
    }
    params = {};
    params.headers = [];
    if (args.headers != null) {
      params.headers = args.headers;
      delete args.headers;
    }

    var possibleParams = [];
    for (var i = 0; i < this.parameters.length; i++) {
      var param = this.parameters[i];
      if (param.paramType === 'header') {
        if (args[param.name])
          params.headers[param.name] = args[param.name];
      }
      else if (param.paramType === 'form' || param.paramType.toLowerCase() === 'file')
        possibleParams.push(param);
      else if (param.paramType === 'body' && param.name !== 'body') {
        if (args.body) {
          throw new Error("Saw two body params in an API listing; expecting a max of one.");
        }
        args.body = args[param.name];
      }
    }

    if (args.body != null) {
      params.body = args.body;
      delete args.body;
    }

    if (possibleParams) {
      var key;
      for (key in possibleParams) {
        var value = possibleParams[key];
        if (args[value.name]) {
          params[value.name] = args[value.name];
        }
      }
    }

    req = new SwaggerRequest(this.method, this.urlify(args), params, opts, callback, error, this);
    if (opts.mock != null) {
      return req;
    } else {
      return true;
    }
  };

  SwaggerOperation.prototype.pathJson = function () {
    return this.path.replace("{format}", "json");
  };

  SwaggerOperation.prototype.pathXml = function () {
    return this.path.replace("{format}", "xml");
  };

  SwaggerOperation.prototype.encodePathParam = function (pathParam) {
    var encParts, part, parts, _i, _len;
    pathParam = pathParam.toString();
    if (pathParam.indexOf("/") === -1) {
      return encodeURIComponent(pathParam);
    } else {
      parts = pathParam.split("/");
      encParts = [];
      for (_i = 0, _len = parts.length; _i < _len; _i++) {
        part = parts[_i];
        encParts.push(encodeURIComponent(part));
      }
      return encParts.join("/");
    }
  };

  SwaggerOperation.prototype.urlify = function (args) {
    var url = this.resource.basePath + this.pathJson();
    var params = this.parameters;
    for (var i = 0; i < params.length; i++) {
      var param = params[i];
      if (param.paramType === 'path') {
        if (args[param.name]) {
          // apply path params and remove from args
          var reg = new RegExp('\\{\\s*?' + param.name + '.*?\\}(?=\\s*?(\\/?|$))', 'gi');
          url = url.replace(reg, this.encodePathParam(args[param.name]));
          delete args[param.name];
        }
        else
          throw "" + param.name + " is a required path param.";
      }
    }

    var queryParams = "";
    for (var i = 0; i < params.length; i++) {
      var param = params[i];
      if(param.paramType === 'query') {
        if (queryParams !== '')
          queryParams += '&';    
        if (Array.isArray(param)) {
          var j;   
          var output = '';   
          for(j = 0; j < param.length; j++) {    
            if(j > 0)    
              output += ',';   
            output += encodeURIComponent(param[j]);    
          }    
          queryParams += encodeURIComponent(param.name) + '=' + output;    
        }
        else {
          if (args[param.name]) {
            queryParams += encodeURIComponent(param.name) + '=' + encodeURIComponent(args[param.name]);
          } else {
            if (param.required)
              throw "" + param.name + " is a required query param.";
          }
        }
      }
    }
    if ((queryParams != null) && queryParams.length > 0)
      url += '?' + queryParams;
    return url;
  };

  SwaggerOperation.prototype.supportHeaderParams = function () {
    return this.resource.api.supportHeaderParams;
  };

  SwaggerOperation.prototype.supportedSubmitMethods = function () {
    return this.resource.api.supportedSubmitMethods;
  };

  SwaggerOperation.prototype.getQueryParams = function (args) {
    return this.getMatchingParams(['query'], args);
  };

  SwaggerOperation.prototype.getHeaderParams = function (args) {
    return this.getMatchingParams(['header'], args);
  };

  SwaggerOperation.prototype.getMatchingParams = function (paramTypes, args) {
    var matchingParams = {};
    var params = this.parameters;
    for (var i = 0; i < params.length; i++) {
      param = params[i];
      if (args && args[param.name])
        matchingParams[param.name] = args[param.name];
    }
    var headers = this.resource.api.headers;
    var name;
    for (name in headers) {
      var value = headers[name];
      matchingParams[name] = value;
    }
    return matchingParams;
  };

  SwaggerOperation.prototype.help = function () {
    var msg = "";
    var params = this.parameters;
    for (var i = 0; i < params.length; i++) {
      var param = params[i];
      if (msg !== "")
        msg += "\n";
      msg += "* " + param.name + (param.required ? ' (required)' : '') + " - " + param.description;
    }
    return msg;
  };


  SwaggerOperation.prototype.formatXml = function (xml) {
    var contexp, formatted, indent, lastType, lines, ln, pad, reg, transitions, wsexp, _fn, _i, _len;
    reg = /(>)(<)(\/*)/g;
    wsexp = /[ ]*(.*)[ ]+\n/g;
    contexp = /(<.+>)(.+\n)/g;
    xml = xml.replace(reg, '$1\n$2$3').replace(wsexp, '$1\n').replace(contexp, '$1\n$2');
    pad = 0;
    formatted = '';
    lines = xml.split('\n');
    indent = 0;
    lastType = 'other';
    transitions = {
      'single->single': 0,
      'single->closing': -1,
      'single->opening': 0,
      'single->other': 0,
      'closing->single': 0,
      'closing->closing': -1,
      'closing->opening': 0,
      'closing->other': 0,
      'opening->single': 1,
      'opening->closing': 0,
      'opening->opening': 1,
      'opening->other': 1,
      'other->single': 0,
      'other->closing': -1,
      'other->opening': 0,
      'other->other': 0
    };
    _fn = function (ln) {
      var fromTo, j, key, padding, type, types, value;
      types = {
        single: Boolean(ln.match(/<.+\/>/)),
        closing: Boolean(ln.match(/<\/.+>/)),
        opening: Boolean(ln.match(/<[^!?].*>/))
      };
      type = ((function () {
        var _results;
        _results = [];
        for (key in types) {
          value = types[key];
          if (value) {
            _results.push(key);
          }
        }
        return _results;
      })())[0];
      type = type === void 0 ? 'other' : type;
      fromTo = lastType + '->' + type;
      lastType = type;
      padding = '';
      indent += transitions[fromTo];
      padding = ((function () {
        var _j, _ref5, _results;
        _results = [];
        for (j = _j = 0, _ref5 = indent; 0 <= _ref5 ? _j < _ref5 : _j > _ref5; j = 0 <= _ref5 ? ++_j : --_j) {
          _results.push('  ');
        }
        return _results;
      })()).join('');
      if (fromTo === 'opening->closing') {
        return formatted = formatted.substr(0, formatted.length - 1) + ln + '\n';
      } else {
        return formatted += padding + ln + '\n';
      }
    };
    for (_i = 0, _len = lines.length; _i < _len; _i++) {
      ln = lines[_i];
      _fn(ln);
    }
    return formatted;
  };

  var SwaggerRequest = function (type, url, params, opts, successCallback, errorCallback, operation, execution) {
    var _this = this;
    var errors = [];
    this.useJQuery = (typeof operation.resource.useJQuery !== 'undefined' ? operation.resource.useJQuery : null);
    this.type = (type || errors.push("SwaggerRequest type is required (get/post/put/delete/patch/options)."));
    this.url = (url || errors.push("SwaggerRequest url is required."));
    this.params = params;
    this.opts = opts;
    this.successCallback = (successCallback || errors.push("SwaggerRequest successCallback is required."));
    this.errorCallback = (errorCallback || errors.push("SwaggerRequest error callback is required."));
    this.operation = (operation || errors.push("SwaggerRequest operation is required."));
    this.execution = execution;
    this.headers = (params.headers || {});

    if (errors.length > 0) {
      throw errors;
    }

    this.type = this.type.toUpperCase();

    // set request, response content type headers
    var headers = this.setHeaders(params, this.operation);
    var body = params.body;

    // encode the body for form submits
    if (headers["Content-Type"]) {
      var values = {};
      var i;
      var operationParams = this.operation.parameters;
      for (i = 0; i < operationParams.length; i++) {
        var param = operationParams[i];
        if (param.paramType === "form")
          values[param.name] = param;
      }

      if (headers["Content-Type"].indexOf("application/x-www-form-urlencoded") === 0) {
        var encoded = "";
        var key, value;
        for (key in values) {
          value = this.params[key];
          if (typeof value !== 'undefined') {
            if (encoded !== "")
              encoded += "&";
            encoded += encodeURIComponent(key) + '=' + encodeURIComponent(value);
          }
        }
        body = encoded;
      }
      else if (headers["Content-Type"].indexOf("multipart/form-data") === 0) {
        // encode the body for form submits
        var data = "";
        var boundary = "----SwaggerFormBoundary" + Date.now();
        var key, value;
        for (key in values) {
          value = this.params[key];
          if (typeof value !== 'undefined') {
            data += '--' + boundary + '\n';
            data += 'Content-Disposition: form-data; name="' + key + '"';
            data += '\n\n';
            data += value + "\n";
          }
        }
        data += "--" + boundary + "--\n";
        headers["Content-Type"] = "multipart/form-data; boundary=" + boundary;
        body = data;
      }
    }

    var obj;
    if (!((this.headers != null) && (this.headers.mock != null))) {
      obj = {
        url: this.url,
        method: this.type,
        headers: headers,
        body: body,
        useJQuery: this.useJQuery,
        on: {
          error: function (response) {
            return _this.errorCallback(response, _this.opts.parent);
          },
          redirect: function (response) {
            return _this.successCallback(response, _this.opts.parent);
          },
          307: function (response) {
            return _this.successCallback(response, _this.opts.parent);
          },
          response: function (response) {
            return _this.successCallback(response, _this.opts.parent);
          }
        }
      };
      var e;
      if (typeof window !== 'undefined') {
        e = window;
      } else {
        e = exports;
      }
      var status = e.authorizations.apply(obj, this.operation.authorizations);
      if (opts.mock == null) {
        if (status !== false) {
          new SwaggerHttp().execute(obj);
        } else {
          obj.canceled = true;
        }
      } else {
        return obj;
      }
    }
    return obj;
  };

  SwaggerRequest.prototype.setHeaders = function (params, operation) {
    // default type
    var accepts = "application/json";
    var consumes = "application/json";

    var allDefinedParams = this.operation.parameters;
    var definedFormParams = [];
    var definedFileParams = [];
    var body = params.body;
    var headers = {};

    // get params from the operation and set them in definedFileParams, definedFormParams, headers
    var i;
    for (i = 0; i < allDefinedParams.length; i++) {
      var param = allDefinedParams[i];
      if (param.paramType === "form")
        definedFormParams.push(param);
      else if (param.paramType === "file")
        definedFileParams.push(param);
      else if (param.paramType === "header" && this.params.headers) {
        var key = param.name;
        var headerValue = this.params.headers[param.name];
        if (typeof this.params.headers[param.name] !== 'undefined')
          headers[key] = headerValue;
      }
    }

    // if there's a body, need to set the accepts header via requestContentType
    if (body && (this.type === "POST" || this.type === "PUT" || this.type === "PATCH" || this.type === "DELETE")) {
      if (this.opts.requestContentType)
        consumes = this.opts.requestContentType;
    } else {
      // if any form params, content type must be set
      if (definedFormParams.length > 0) {
        if (definedFileParams.length > 0)
          consumes = "multipart/form-data";
        else
          consumes = "application/x-www-form-urlencoded";
      }
      else if (this.type === "DELETE")
        body = "{}";
      else if (this.type != "DELETE")
        consumes = null;
    }

    if (consumes && this.operation.consumes) {
      if (this.operation.consumes.indexOf(consumes) === -1) {
        log("server doesn't consume " + consumes + ", try " + JSON.stringify(this.operation.consumes));
        consumes = this.operation.consumes[0];
      }
    }

    if (this.opts.responseContentType) {
      accepts = this.opts.responseContentType;
    } else {
      accepts = "application/json";
    }
    if (accepts && this.operation.produces) {
      if (this.operation.produces.indexOf(accepts) === -1) {
        log("server can't produce " + accepts);
        accepts = this.operation.produces[0];
      }
    }

    if ((consumes && body !== "") || (consumes === "application/x-www-form-urlencoded"))
      headers["Content-Type"] = consumes;
    if (accepts)
      headers["Accept"] = accepts;
    return headers;
  }

  SwaggerRequest.prototype.asCurl = function () {
    var results = [];
    if (this.headers) {
      var key;
      for (key in this.headers) {
        results.push("--header \"" + key + ": " + this.headers[v] + "\"");
      }
    }
    return "curl " + (results.join(" ")) + " " + this.url;
  };

  /**
   * SwaggerHttp is a wrapper for executing requests
   */
  var SwaggerHttp = function () { };

  SwaggerHttp.prototype.execute = function (obj) {
    if (obj && (typeof obj.useJQuery === 'boolean'))
      this.useJQuery = obj.useJQuery;
    else
      this.useJQuery = this.isIE8();

    if (this.useJQuery)
      return new JQueryHttpClient().execute(obj);
    else
      return new ShredHttpClient().execute(obj);
  }

  SwaggerHttp.prototype.isIE8 = function () {
    var detectedIE = false;
    if (typeof navigator !== 'undefined' && navigator.userAgent) {
      nav = navigator.userAgent.toLowerCase();
      if (nav.indexOf('msie') !== -1) {
        var version = parseInt(nav.split('msie')[1]);
        if (version <= 8) {
          detectedIE = true;
        }
      }
    }
    return detectedIE;
  };

  /*
   * JQueryHttpClient lets a browser take advantage of JQuery's cross-browser magic.
   * NOTE: when jQuery is available it will export both '$' and 'jQuery' to the global space.
   *     Since we are using closures here we need to alias it for internal use.
   */
  var JQueryHttpClient = function (options) {
    "use strict";
    if (!jQuery) {
      var jQuery = window.jQuery;
    }
  }

  JQueryHttpClient.prototype.execute = function (obj) {
    var cb = obj.on;
    var request = obj;

    obj.type = obj.method;
    obj.cache = false;

    obj.beforeSend = function (xhr) {
      var key, results;
      if (obj.headers) {
        results = [];
        var key;
        for (key in obj.headers) {
          if (key.toLowerCase() === "content-type") {
            results.push(obj.contentType = obj.headers[key]);
          } else if (key.toLowerCase() === "accept") {
            results.push(obj.accepts = obj.headers[key]);
          } else {
            results.push(xhr.setRequestHeader(key, obj.headers[key]));
          }
        }
        return results;
      }
    };

    obj.data = obj.body;
    obj.complete = function (response, textStatus, opts) {
      var headers = {},
        headerArray = response.getAllResponseHeaders().split("\n");

      for (var i = 0; i < headerArray.length; i++) {
        var toSplit = headerArray[i].trim();
        if (toSplit.length === 0)
          continue;
        var separator = toSplit.indexOf(":");
        if (separator === -1) {
          // Name but no value in the header
          headers[toSplit] = null;
          continue;
        }
        var name = toSplit.substring(0, separator).trim(),
          value = toSplit.substring(separator + 1).trim();
        headers[name] = value;
      }

      var out = {
        url: request.url,
        method: request.method,
        status: response.status,
        data: response.responseText,
        headers: headers
      };

      var contentType = (headers["content-type"] || headers["Content-Type"] || null)

      if (contentType != null) {
        if (contentType.indexOf("application/json") == 0 || contentType.indexOf("+json") > 0) {
          if (response.responseText && response.responseText !== "")
            out.obj = JSON.parse(response.responseText);
          else
            out.obj = {}
        }
      }

      if (response.status >= 200 && response.status < 300)
        cb.response(out);
      else if (response.status === 0 || (response.status >= 400 && response.status < 599))
        cb.error(out);
      else
        return cb.response(out);
    };

    jQuery.support.cors = true;
    return jQuery.ajax(obj);
  }

  /*
   * ShredHttpClient is a light-weight, node or browser HTTP client
   */
  var ShredHttpClient = function (options) {
    this.options = (options || {});
    this.isInitialized = false;

    var identity, toString;

    if (typeof window !== 'undefined') {
      this.Shred = require("./shred");
      this.content = require("./shred/content");
    }
    else
      this.Shred = require("shred");
    this.shred = new this.Shred();
  }

  ShredHttpClient.prototype.initShred = function () {
    this.isInitialized = true;
    this.registerProcessors(this.shred);
  }

  ShredHttpClient.prototype.registerProcessors = function (shred) {
    var identity = function (x) {
      return x;
    };
    var toString = function (x) {
      return x.toString();
    };

    if (typeof window !== 'undefined') {
      this.content.registerProcessor(["application/json; charset=utf-8", "application/json", "json"], {
        parser: identity,
        stringify: toString
      });
    } else {
      this.Shred.registerProcessor(["application/json; charset=utf-8", "application/json", "json"], {
        parser: identity,
        stringify: toString
      });
    }
  }

  ShredHttpClient.prototype.execute = function (obj) {
    if (!this.isInitialized)
      this.initShred();

    var cb = obj.on, res;

    var transform = function (response) {
      var out = {
        headers: response._headers,
        url: response.request.url,
        method: response.request.method,
        status: response.status,
        data: response.content.data
      };

      var headers = response._headers.normalized || response._headers;
      var contentType = (headers["content-type"] || headers["Content-Type"] || null)
      if (contentType != null) {
        if (contentType.indexOf("application/json") == 0 || contentType.indexOf("+json") > 0) {
          if (response.content.data && response.content.data !== "")
            try {
              out.obj = JSON.parse(response.content.data);
            }
            catch (ex) {
              // do not set out.obj
              log ("unable to parse JSON content");
            }
          else
            out.obj = {}
        }
      }
      return out;
    };

    // Transform an error into a usable response-like object
    var transformError = function (error) {
      var out = {
        // Default to a status of 0 - The client will treat this as a generic permissions sort of error
        status: 0,
        data: error.message || error
      };

      if (error.code) {
        out.obj = error;

        if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
          // We can tell the client that this should be treated as a missing resource and not as a permissions thing
          out.status = 404;
        }
      }

      return out;
    };

    var res = {
      error: function (response) {
        if (obj)
          return cb.error(transform(response));
      },
      // Catch the Shred error raised when the request errors as it is made (i.e. No Response is coming)
      request_error: function (err) {
        if (obj)
          return cb.error(transformError(err));
      },
      redirect: function (response) {
        if (obj)
          return cb.redirect(transform(response));
      },
      307: function (response) {
        if (obj)
          return cb.redirect(transform(response));
      },
      response: function (response) {
        if (obj)
          return cb.response(transform(response));
      }
    };
    if (obj) {
      obj.on = res;
    }
    return this.shred.request(obj);
  };

  /**
   * SwaggerAuthorizations applys the correct authorization to an operation being executed
   */
  var SwaggerAuthorizations = function () {
    this.authz = {};
  };

  SwaggerAuthorizations.prototype.add = function (name, auth) {
    this.authz[name] = auth;
    return auth;
  };

  SwaggerAuthorizations.prototype.remove = function (name) {
    return delete this.authz[name];
  };

  SwaggerAuthorizations.prototype.apply = function (obj, authorizations) {
    var status = null;
    var key, value, result;

    // if the "authorizations" key is undefined, or has an empty array, add all keys
    if (typeof authorizations === 'undefined' || Object.keys(authorizations).length == 0) {
      for (key in this.authz) {
        value = this.authz[key];
        result = value.apply(obj, authorizations);
        if (result === true)
          status = true;
      }
    }
    else {
      for (name in authorizations) {
        for (key in this.authz) {
          if (key == name) {
            value = this.authz[key];
            result = value.apply(obj, authorizations);
            if (result === true)
              status = true;
          }
        }
      }
    }

    return status;
  };

  /**
   * ApiKeyAuthorization allows a query param or header to be injected
   */
  var ApiKeyAuthorization = function (name, value, type, delimiter) {
    this.name = name;
    this.value = value;
    this.type = type;
    this.delimiter = delimiter;
  };

  ApiKeyAuthorization.prototype.apply = function (obj, authorizations) {
    if (this.type === "query") {
      if (obj.url.indexOf('?') > 0)
        obj.url = obj.url + "&" + this.name + "=" + this.value;
      else
        obj.url = obj.url + "?" + this.name + "=" + this.value;
      return true;
    } else if (this.type === "header") {
      if (typeof obj.headers[this.name] !== 'undefined') {
        if (typeof this.delimiter !== 'undefined')
          obj.headers[this.name] = obj.headers[this.name] + this.delimiter + this.value;
      }
      else
        obj.headers[this.name] = this.value;
      return true;
    }
  };

  var CookieAuthorization = function (cookie) {
    this.cookie = cookie;
  }

  CookieAuthorization.prototype.apply = function (obj, authorizations) {
    obj.cookieJar = obj.cookieJar || CookieJar();
    obj.cookieJar.setCookie(this.cookie);
    return true;
  }

  /**
   * Password Authorization is a basic auth implementation
   */
  var PasswordAuthorization = function (name, username, password) {
    this.name = name;
    this.username = username;
    this.password = password;
    this._btoa = null;
    if (typeof window !== 'undefined')
      this._btoa = btoa;
    else
      this._btoa = require("btoa");
  };

  PasswordAuthorization.prototype.apply = function (obj, authorizations) {
    var base64encoder = this._btoa;
    obj.headers["Authorization"] = "Basic " + base64encoder(this.username + ":" + this.password);
    return true;
  };

  var e = (typeof window !== 'undefined' ? window : exports);

  var sampleModels = {};
  var cookies = {};

  e.parameterMacro = parameterMacro;
  e.modelPropertyMacro = modelPropertyMacro;
  e.SampleModels = sampleModels;
  e.SwaggerHttp = SwaggerHttp;
  e.SwaggerRequest = SwaggerRequest;
  e.authorizations = new SwaggerAuthorizations();
  e.ApiKeyAuthorization = ApiKeyAuthorization;
  e.PasswordAuthorization = PasswordAuthorization;
  e.CookieAuthorization = CookieAuthorization;
  e.JQueryHttpClient = JQueryHttpClient;
  e.ShredHttpClient = ShredHttpClient;
  e.SwaggerOperation = SwaggerOperation;
  e.SwaggerModel = SwaggerModel;
  e.SwaggerModelProperty = SwaggerModelProperty;
  e.SwaggerResource = SwaggerResource;
  e.SwaggerApi = SwaggerApi;
  e.log = log;

})();
