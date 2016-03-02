/**
 * swagger-client - swagger.js is a javascript client for use with swaggering APIs.
 * @version v2.1.1-M1
 * @link http://swagger.io
 * @license apache 2.0
 */
(function(){
var ArrayModel = function(definition) {
  this.name = "arrayModel";
  this.definition = definition || {};
  this.properties = [];
  
  var requiredFields = definition.enum || [];
  var innerType = definition.items;
  if(innerType) {
    if(innerType.type) {
      this.type = typeFromJsonSchema(innerType.type, innerType.format);
    }
    else {
      this.ref = innerType.$ref;
    }
  }
  return this;
};

ArrayModel.prototype.createJSONSample = function(modelsToIgnore) {
  var result;
  modelsToIgnore = (modelsToIgnore||{});
  if(this.type) {
    result = this.type;
  }
  else if (this.ref) {
    var name = simpleRef(this.ref);
    if(typeof modelsToIgnore[name] === 'undefined') {
      modelsToIgnore[name] = this;
      result = models[name].createJSONSample(modelsToIgnore);
    }
    else {
      return name;
    }
  }
  return [ result ];
};

ArrayModel.prototype.getSampleValue = function(modelsToIgnore) {
  var result;
  modelsToIgnore = (modelsToIgnore || {});
  if(this.type) {
    result = type;
  }
  else if (this.ref) {
    var name = simpleRef(this.ref);
    result = models[name].getSampleValue(modelsToIgnore);
  }
  return [ result ];
};

ArrayModel.prototype.getMockSignature = function(modelsToIgnore) {
  var propertiesStr = [];

  if(this.ref) {
    return models[simpleRef(this.ref)].getMockSignature();
  }
};


/**
 * SwaggerAuthorizations applys the correct authorization to an operation being executed
 */
var SwaggerAuthorizations = function() {
  this.authz = {};
};

SwaggerAuthorizations.prototype.add = function(name, auth) {
  this.authz[name] = auth;
  return auth;
};

SwaggerAuthorizations.prototype.remove = function(name) {
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
    // 2.0 support
    if (Array.isArray(authorizations)) {

      for (var i = 0; i < authorizations.length; i++) {
        var auth = authorizations[i];
        for (name in auth) {
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
    }
    else {
      // 1.2 support
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
  }

  return status;
};

/**
 * ApiKeyAuthorization allows a query param or header to be injected
 */
var ApiKeyAuthorization = function(name, value, type) {
  this.name = name;
  this.value = value;
  this.type = type;
};

ApiKeyAuthorization.prototype.apply = function(obj, authorizations) {
  if (this.type === "query") {
    if (obj.url.indexOf('?') > 0)
      obj.url = obj.url + "&" + this.name + "=" + this.value;
    else
      obj.url = obj.url + "?" + this.name + "=" + this.value;
    return true;
  } else if (this.type === "header") {
    obj.headers[this.name] = this.value;
    return true;
  }
};

var CookieAuthorization = function(cookie) {
  this.cookie = cookie;
};

CookieAuthorization.prototype.apply = function(obj, authorizations) {
  obj.cookieJar = obj.cookieJar || CookieJar();
  obj.cookieJar.setCookie(this.cookie);
  return true;
};

/**
 * Password Authorization is a basic auth implementation
 */
var PasswordAuthorization = function(name, username, password) {
  this.name = name;
  this.username = username;
  this.password = password;
  this._btoa = null;
  if (typeof window !== 'undefined')
    this._btoa = btoa;
  else
    this._btoa = require("btoa");
};

PasswordAuthorization.prototype.apply = function(obj, authorizations) {
  var base64encoder = this._btoa;
  obj.headers.Authorization = "Basic " + base64encoder(this.username + ":" + this.password);
  return true;
};
var __bind = function(fn, me){
  return function(){
    return fn.apply(me, arguments);
  };
};

fail = function(message) {
  log(message);
};

log = function(){
  log.history = log.history || [];
  log.history.push(arguments);
  if(this.console){
    console.log( Array.prototype.slice.call(arguments)[0] );
  }
};

if (!Array.prototype.indexOf) {
  Array.prototype.indexOf = function(obj, start) {
    for (var i = (start || 0), j = this.length; i < j; i++) {
      if (this[i] === obj) { return i; }
    }
    return -1;
  };
}

/**
 * allows override of the default value based on the parameter being
 * supplied
 **/
var applyParameterMacro = function (operation, parameter) {
  var e = (typeof window !== 'undefined' ? window : exports);
  if(e.parameterMacro)
    return e.parameterMacro(operation, parameter);
  else
    return parameter.defaultValue;
};

/**
 * allows overriding the default value of an model property
 **/
var applyModelPropertyMacro = function (model, property) {
  var e = (typeof window !== 'undefined' ? window : exports);
  if(e.modelPropertyMacro)
    return e.modelPropertyMacro(model, property);
  else
    return property.defaultValue;
};

/**
 * PrimitiveModel
 **/
var PrimitiveModel = function(definition) {
  this.name = "name";
  this.definition = definition || {};
  this.properties = [];

  var requiredFields = definition.enum || [];
  this.type = typeFromJsonSchema(definition.type, definition.format);
};

PrimitiveModel.prototype.createJSONSample = function(modelsToIgnore) {
  var result = this.type;
  return result;
};

PrimitiveModel.prototype.getSampleValue = function() {
  var result = this.type;
  return null;
};

PrimitiveModel.prototype.getMockSignature = function(modelsToIgnore) {
  var propertiesStr = [];
  var i, prop;
  for (i = 0; i < this.properties.length; i++) {
    prop = this.properties[i];
    propertiesStr.push(prop.toString());
  }

  var strong = '<span class="strong">';
  var stronger = '<span class="stronger">';
  var strongClose = '</span>';
  var classOpen = strong + this.name + ' {' + strongClose;
  var classClose = strong + '}' + strongClose;
  var returnVal = classOpen + '<div>' + propertiesStr.join(',</div><div>') + '</div>' + classClose;

  if (!modelsToIgnore)
    modelsToIgnore = {};
  modelsToIgnore[this.name] = this;
  for (i = 0; i < this.properties.length; i++) {
    prop = this.properties[i];
    var ref = prop.$ref;
    var model = models[ref];
    if (model && typeof modelsToIgnore[ref] === 'undefined') {
      returnVal = returnVal + ('<br>' + model.getMockSignature(modelsToIgnore));
    }
  }
  return returnVal;
};
var SwaggerClient = function(url, options) {
  this.isBuilt = false;
  this.url = null;
  this.debug = false;
  this.basePath = null;
  this.modelsArray = [];
  this.authorizations = null;
  this.authorizationScheme = null;
  this.isValid = false;
  this.info = null;
  this.useJQuery = false;

  if(typeof url !== 'undefined')
    return this.initialize(url, options);
};

SwaggerClient.prototype.initialize = function (url, options) {
  this.models = models;

  options = (options||{});

  if(typeof url === 'string')
    this.url = url;
  else if(typeof url === 'object') {
    options = url;
    this.url = options.url;
  }
  this.swaggerRequstHeaders = options.swaggerRequstHeaders || 'application/json;charset=utf-8,*/*';
  this.defaultSuccessCallback = options.defaultSuccessCallback || null;
  this.defaultErrorCallback = options.defaultErrorCallback || null;

  if (typeof options.success === 'function')
    this.success = options.success;

  if (options.useJQuery)
    this.useJQuery = options.useJQuery;

  if (options.authorizations) {
    this.clientAuthorizations = options.authorizations;
  } else {
    var e = (typeof window !== 'undefined' ? window : exports);
    this.clientAuthorizations = e.authorizations;
  }

  this.supportedSubmitMethods = options.supportedSubmitMethods || [];
  this.failure = options.failure || function() {};
  this.progress = options.progress || function() {};
  this.spec = options.spec;
  this.options = options;

  if (typeof options.success === 'function') {
    this.build();
  }
};

SwaggerClient.prototype.build = function(mock) {
  if (this.isBuilt) return this;
  var self = this;
  this.progress('fetching resource list: ' + this.url);
  var obj = {
    useJQuery: this.useJQuery,
    url: this.url,
    method: "get",
    headers: {
      accept: this.swaggerRequstHeaders
    },
    on: {
      error: function(response) {
        if (self.url.substring(0, 4) !== 'http')
          return self.fail('Please specify the protocol for ' + self.url);
        else if (response.status === 0)
          return self.fail('Can\'t read from server.  It may not have the appropriate access-control-origin settings.');
        else if (response.status === 404)
          return self.fail('Can\'t read swagger JSON from ' + self.url);
        else
          return self.fail(response.status + ' : ' + response.statusText + ' ' + self.url);
      },
      response: function(resp) {
        var responseObj = resp.obj || JSON.parse(resp.data);
        self.swaggerVersion = responseObj.swaggerVersion;

        if(responseObj.swagger && parseInt(responseObj.swagger) === 2) {
          self.swaggerVersion = responseObj.swagger;
          self.buildFromSpec(responseObj);
          self.isValid = true;
        }
        else {
          if (self.swaggerVersion === '1.2') {
            return self.buildFrom1_2Spec(responseObj);
          } else {
            return self.buildFrom1_1Spec(responseObj);
          }
        }
      }
    }
  };
  if(this.spec) {
    setTimeout(function() { self.buildFromSpec(self.spec); }, 10);
  }
  else {
    var e = (typeof window !== 'undefined' ? window : exports);
    var status = e.authorizations.apply(obj);
    if(mock)
      return obj;
    new SwaggerHttp().execute(obj);
  }

  return this;
};

SwaggerClient.prototype.buildFromSpec = function(response) {
  if(this.isBuilt) return this;

  this.info = response.info || {};
  this.title = response.title || '';
  this.host = response.host || '';
  this.schemes = response.schemes || [];
  this.basePath = response.basePath || '';
  this.apis = {};
  this.apisArray = [];
  this.consumes = response.consumes;
  this.produces = response.produces;
  this.securityDefinitions = response.securityDefinitions;

  // legacy support
  this.authSchemes = response.securityDefinitions;

  var location;

  if(typeof this.url === 'string') {
    location = this.parseUri(this.url);
  }

  if(typeof this.schemes === 'undefined' || this.schemes.length === 0) {
    this.scheme = location.scheme || 'http';
  }
  else {
    this.scheme = this.schemes[0];
  }

  if(typeof this.host === 'undefined' || this.host === '') {
    this.host = location.host;
    if (location.port) {
      this.host = this.host + ':' + location.port;
    }
  }

  this.definitions = response.definitions;
  var key;
  for(key in this.definitions) {
    var model = new Model(key, this.definitions[key]);
    if(model) {
      models[key] = model;
    }
  }

  // get paths, create functions for each operationId
  var path;
  var operations = [];
  for(path in response.paths) {
    if(typeof response.paths[path] === 'object') {
      var httpMethod;
      for(httpMethod in response.paths[path]) {
        if(['delete', 'get', 'head', 'options', 'patch', 'post', 'put'].indexOf(httpMethod) === -1) {
          continue;
        }
        var operation = response.paths[path][httpMethod];
        var tags = operation.tags;
        if(typeof tags === 'undefined') {
          operation.tags = [ 'default' ];
          tags = operation.tags;
        }
        var operationId = this.idFromOp(path, httpMethod, operation);
        var operationObject = new Operation (
          this,
          operation.scheme,
          operationId,
          httpMethod,
          path,
          operation,
          this.definitions
        );
        // bind this operation's execute command to the api
        if(tags.length > 0) {
          var i;
          for(i = 0; i < tags.length; i++) {
            var tag = this.tagFromLabel(tags[i]);
            var operationGroup = this[tag];
            if(typeof operationGroup === 'undefined') {
              this[tag] = [];
              operationGroup = this[tag];
              operationGroup.operations = {};
              operationGroup.label = tag;
              operationGroup.apis = [];
              this[tag].help = this.help.bind(operationGroup);
              this.apisArray.push(new OperationGroup(tag, operationObject));
            }
            operationGroup[operationId] = operationObject.execute.bind(operationObject);
            operationGroup[operationId].help = operationObject.help.bind(operationObject);
            operationGroup.apis.push(operationObject);
            operationGroup.operations[operationId] = operationObject;

            // legacy UI feature
            var j;
            var api;
            for(j = 0; j < this.apisArray.length; j++) {
              if(this.apisArray[j].tag === tag) {
                api = this.apisArray[j];
              }
            }
            if(api) {
              api.operationsArray.push(operationObject);
            }
          }
        }
        else {
          log('no group to bind to');
        }
      }
    }
  }
  this.isBuilt = true;
  if (this.success)
    this.success();
  return this;
};

SwaggerClient.prototype.parseUri = function(uri) {
  var urlParseRE = /^(((([^:\/#\?]+:)?(?:(\/\/)((?:(([^:@\/#\?]+)(?:\:([^:@\/#\?]+))?)@)?(([^:\/#\?\]\[]+|\[[^\/\]@#?]+\])(?:\:([0-9]+))?))?)?)?((\/?(?:[^\/\?#]+\/+)*)([^\?#]*)))?(\?[^#]+)?)(#.*)?/;
  var parts = urlParseRE.exec(uri);
  return {
    scheme: parts[4].replace(':',''),
    host: parts[11],
    port: parts[12],
    path: parts[15]
  };
};

SwaggerClient.prototype.help = function() {
  var i;
  log('operations for the "' + this.label + '" tag');
  for(i = 0; i < this.apis.length; i++) {
    var api = this.apis[i];
    log('  * ' + api.nickname + ': ' + api.operation.summary);
  }
};

SwaggerClient.prototype.tagFromLabel = function(label) {
  return label;
};

SwaggerClient.prototype.idFromOp = function(path, httpMethod, op) {
  var opId = op.operationId || (path.substring(1) + '_' + httpMethod);
  return opId.replace(/[\.,-\/#!$%\^&\*;:{}=\-_`~()\+\s]/g,'_');
};

SwaggerClient.prototype.fail = function(message) {
  this.failure(message);
  throw message;
};

var OperationGroup = function(tag, operation) {
  this.tag = tag;
  this.path = tag;
  this.name = tag;
  this.operation = operation;
  this.operationsArray = [];

  this.description = operation.description || "";
};

var Operation = function(parent, scheme, operationId, httpMethod, path, args, definitions) {
  var errors = [];
  parent = parent||{};
  args = args||{};

  this.operations = {};
  this.operation = args;
  this.deprecated = args.deprecated;
  this.consumes = args.consumes;
  this.produces = args.produces;
  this.parent = parent;
  this.host = parent.host || 'localhost';
  this.schemes = parent.schemes;
  this.scheme = scheme || parent.scheme || 'http';
  this.basePath = parent.basePath || '/';
  this.nickname = (operationId||errors.push('Operations must have a nickname.'));
  this.method = (httpMethod||errors.push('Operation ' + operationId + ' is missing method.'));
  this.path = (path||errors.push('Operation ' + this.nickname + ' is missing path.'));
  this.parameters = args !== null ? (args.parameters||[]) : {};
  this.summary = args.summary || '';
  this.responses = (args.responses||{});
  this.type = null;
  this.security = args.security;
  this.authorizations = args.security;
  this.description = args.description;
  this.useJQuery = parent.useJQuery;

  if(definitions) {
    // add to global models
    var key;
    for(key in this.definitions) {
      var model = new Model(key, definitions[key]);
      if(model) {
        models[key] = model;
      }
    }
  }

  var i;
  for(i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    if(param.type === 'array') {
      param.isList = true;
      param.allowMultiple = true;
    }
    var innerType = this.getType(param);
    if(innerType && innerType.toString().toLowerCase() === 'boolean') {
      param.allowableValues = {};
      param.isList = true;
      param['enum'] = ["true", "false"];
    }
    if(typeof param['enum'] !== 'undefined') {
      var id;
      param.allowableValues = {};
      param.allowableValues.values = [];
      param.allowableValues.descriptiveValues = [];
      for(id = 0; id < param['enum'].length; id++) {
        var value = param['enum'][id];
        var isDefault = (value === param.default) ? true : false;
        param.allowableValues.values.push(value);
        param.allowableValues.descriptiveValues.push({value : value, isDefault: isDefault});
      }
    }
    if(param.type === 'array') {
      innerType = [innerType];
      if(typeof param.allowableValues === 'undefined') {
        // can't show as a list if no values to select from
        delete param.isList;
        delete param.allowMultiple;
      }
    }
    param.signature = this.getModelSignature(innerType, models).toString();
    param.sampleJSON = this.getModelSampleJSON(innerType, models);
    param.responseClassSignature = param.signature;
  }

  var defaultResponseCode, response, model, responses = this.responses;

  if(responses['200']) {
    response = responses['200'];
    defaultResponseCode = '200';
  }
  else if(responses['201']) {
    response = responses['201'];
    defaultResponseCode = '201';
  }
  else if(responses['202']) {
    response = responses['202'];
    defaultResponseCode = '202';
  }
  else if(responses['203']) {
    response = responses['203'];
    defaultResponseCode = '203';
  }
  else if(responses['204']) {
    response = responses['204'];
    defaultResponseCode = '204';
  }
  else if(responses['205']) {
    response = responses['205'];
    defaultResponseCode = '205';
  }
  else if(responses['206']) {
    response = responses['206'];
    defaultResponseCode = '206';
  }
  else if(responses['default']) {
    response = responses['default'];
    defaultResponseCode = 'default';
  }

  if(response && response.schema) {
    var resolvedModel = this.resolveModel(response.schema, definitions);
    delete responses[defaultResponseCode];
    if(resolvedModel) {
      this.successResponse = {};
      this.successResponse[defaultResponseCode] = resolvedModel;
    }
    else {
      this.successResponse = {};
      this.successResponse[defaultResponseCode] = response.schema.type;
    }
    this.type = response;
  }

  if (errors.length > 0) {
    if(this.resource && this.resource.api && this.resource.api.fail)
      this.resource.api.fail(errors);
  }

  return this;
};

OperationGroup.prototype.sort = function(sorter) {

};

Operation.prototype.getType = function (param) {
  var type = param.type;
  var format = param.format;
  var isArray = false;
  var str;
  if(type === 'integer' && format === 'int32')
    str = 'integer';
  else if(type === 'integer' && format === 'int64')
    str = 'long';
  else if(type === 'integer')
    str = 'integer';
  else if(type === 'string' && format === 'date-time')
    str = 'date-time';
  else if(type === 'string' && format === 'date')
    str = 'date';
  else if(type === 'number' && format === 'float')
    str = 'float';
  else if(type === 'number' && format === 'double')
    str = 'double';
  else if(type === 'number')
    str = 'double';
  else if(type === 'boolean')
    str = 'boolean';
  else if(type === 'string')
    str = 'string';
  else if(type === 'array') {
    isArray = true;
    if(param.items)
      str = this.getType(param.items);
  }
  if(param.$ref)
    str = param.$ref;

  var schema = param.schema;
  if(schema) {
    var ref = schema.$ref;
    if(ref) {
      ref = simpleRef(ref);
      if(isArray)
        return [ ref ];
      else
        return ref;
    }
    else
      return this.getType(schema);
  }
  if(isArray)
    return [ str ];
  else
    return str;
};

Operation.prototype.resolveModel = function (schema, definitions) {
  if(typeof schema.$ref !== 'undefined') {
    var ref = schema.$ref;
    if(ref.indexOf('#/definitions/') === 0)
      ref = ref.substring('#/definitions/'.length);
    if(definitions[ref]) {
      return new Model(ref, definitions[ref]);
    }
  }
  if(schema.type === 'array')
    return new ArrayModel(schema);
  else
    return null;
};

Operation.prototype.help = function(dontPrint) {
  var out = this.nickname + ': ' + this.summary + '\n';
  for(var i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    var typeInfo = typeFromJsonSchema(param.type, param.format);
    out += '\n  * ' + param.name + ' (' + typeInfo + '): ' + param.description;
  }
  if(typeof dontPrint === 'undefined')
    log(out);
  return out;
};

Operation.prototype.getModelSignature = function(type, definitions) {
  var isPrimitive, listType;

  if(type instanceof Array) {
    listType = true;
    type = type[0];
  }
  else if(typeof type === 'undefined')
    type = 'undefined';

  if(type === 'string')
    isPrimitive = true;
  else
    isPrimitive = (listType && definitions[listType]) || (definitions[type]) ? false : true;
  if (isPrimitive) {
    if(listType)
      return 'Array[' + type + ']';
    else
      return type.toString();
  } else {
    if (listType)
      return 'Array[' + definitions[type].getMockSignature() + ']';
    else
      return definitions[type].getMockSignature();
  }
};

Operation.prototype.supportHeaderParams = function () {
  return true;
};

Operation.prototype.supportedSubmitMethods = function () {
  return this.parent.supportedSubmitMethods;
};

Operation.prototype.getHeaderParams = function (args) {
  var headers = this.setContentTypes(args, {});
  for(var i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    if(typeof args[param.name] !== 'undefined') {
      if (param.in === 'header') {
        var value = args[param.name];
        if(Array.isArray(value))
          value = this.encodePathCollection(param.collectionFormat, param.name, value);
        else
          value = this.encodePathParam(value);
        headers[param.name] = value;
      }
    }
  }
  return headers;
};

Operation.prototype.urlify = function (args) {
  var formParams = {};
  var requestUrl = this.path;

  // grab params from the args, build the querystring along the way
  var querystring = '';
  for(var i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    if(typeof args[param.name] !== 'undefined') {
      if(param.in === 'path') {
        var reg = new RegExp('\{' + param.name + '\}', 'gi');
        var value = args[param.name];
        if(Array.isArray(value))
          value = this.encodePathCollection(param.collectionFormat, param.name, value);
        else
          value = this.encodePathParam(value);
        requestUrl = requestUrl.replace(reg, value);
      }
      else if (param.in === 'query' && typeof args[param.name] !== 'undefined') {
        if(querystring === '')
          querystring += '?';
        else
          querystring += '&';
        if(typeof param.collectionFormat !== 'undefined') {
          var qp = args[param.name];
          if(Array.isArray(qp))
            querystring += this.encodeQueryCollection(param.collectionFormat, param.name, qp);
          else
            querystring += this.encodeQueryParam(param.name) + '=' + this.encodeQueryParam(args[param.name]);
        }
        else
          querystring += this.encodeQueryParam(param.name) + '=' + this.encodeQueryParam(args[param.name]);
      }
      else if (param.in === 'formData')
        formParams[param.name] = args[param.name];
    }
  }
  var url = this.scheme + '://' + this.host;

  if(this.basePath !== '/')
    url += this.basePath;

  return url + requestUrl + querystring;
};

Operation.prototype.getMissingParams = function(args) {
  var missingParams = [];
  // check required params, track the ones that are missing
  var i;
  for(i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    if(param.required === true) {
      if(typeof args[param.name] === 'undefined')
        missingParams = param.name;
    }
  }
  return missingParams;
};

Operation.prototype.getBody = function(headers, args) {
  var formParams = {};
  var body;

  for(var i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    if(typeof args[param.name] !== 'undefined') {
      if (param.in === 'body') {
        body = args[param.name];
      } else if(param.in === 'formData') {
        formParams[param.name] = args[param.name];
      }
    }
  }

  // handle form params
  if(headers['Content-Type'] === 'application/x-www-form-urlencoded') {
    var encoded = "";
    var key;
    for(key in formParams) {
      value = formParams[key];
      if(typeof value !== 'undefined'){
        if(encoded !== "")
          encoded += "&";
        encoded += encodeURIComponent(key) + '=' + encodeURIComponent(value);
      }
    }
    body = encoded;
  }

  return body;
};

/**
 * gets sample response for a single operation
 **/
Operation.prototype.getModelSampleJSON = function(type, models) {
  var isPrimitive, listType, sampleJson;

  listType = (type instanceof Array);
  isPrimitive = models[type] ? false : true;
  sampleJson = isPrimitive ? void 0 : models[type].createJSONSample();
  if (sampleJson) {
    sampleJson = listType ? [sampleJson] : sampleJson;
    if(typeof sampleJson == 'string')
      return sampleJson;
    else if(typeof sampleJson === 'object') {
      var t = sampleJson;
      if(sampleJson instanceof Array && sampleJson.length > 0) {
        t = sampleJson[0];
      }
      if(t.nodeName) {
        var xmlString = new XMLSerializer().serializeToString(t);
        return this.formatXml(xmlString);
      }
      else
        return JSON.stringify(sampleJson, null, 2);
    }
    else
      return sampleJson;
  }
};

/**
 * legacy binding
 **/
Operation.prototype["do"] = function(args, opts, callback, error, parent) {
  return this.execute(args, opts, callback, error, parent);
};


/**
 * executes an operation
 **/
Operation.prototype.execute = function(arg1, arg2, arg3, arg4, parent) {
  var args = arg1 || {};
  var opts = {}, success, error;
  if(typeof arg2 === 'object') {
    opts = arg2;
    success = arg3;
    error = arg4;
  }

  if(typeof arg2 === 'function') {
    success = arg2;
    error = arg3;
  }

  success = (success||log);
  error = (error||log);

  if(typeof opts.useJQuery === 'boolean') {
    this.useJQuery = opts.useJQuery;
  }

  var missingParams = this.getMissingParams(args);
  if(missingParams.length > 0) {
    var message = 'missing required params: ' + missingParams;
    fail(message);
    return;
  }
  var allHeaders = this.getHeaderParams(args);
  var contentTypeHeaders = this.setContentTypes(args, opts);

  var headers = {};
  for (var attrname in allHeaders) { headers[attrname] = allHeaders[attrname]; }
  for (var attrname in contentTypeHeaders) { headers[attrname] = contentTypeHeaders[attrname]; }

  var body = this.getBody(headers, args);
  var url = this.urlify(args);

  var obj = {
    url: url,
    method: this.method.toUpperCase(),
    body: body,
    useJQuery: this.useJQuery,
    headers: headers,
    on: {
      response: function(response) {
        return success(response, parent);
      },
      error: function(response) {
        return error(response, parent);
      }
    }
  };
  var status = e.authorizations.apply(obj, this.operation.security);
  if(opts.mock === true)
    return obj;
  else
    new SwaggerHttp().execute(obj);
};

Operation.prototype.setContentTypes = function(args, opts) {
  // default type
  var accepts = 'application/json';
  var consumes = args.parameterContentType || 'application/json';

  var allDefinedParams = this.parameters;
  var definedFormParams = [];
  var definedFileParams = [];
  var body;
  var headers = {};

  // get params from the operation and set them in definedFileParams, definedFormParams, headers
  var i;
  for(i = 0; i < allDefinedParams.length; i++) {
    var param = allDefinedParams[i];
    if(param.in === 'formData') {
      if(param.type === 'file')
        definedFileParams.push(param);
      else
        definedFormParams.push(param);
    }
    else if(param.in === 'header' && opts) {
      var key = param.name;
      var headerValue = opts[param.name];
      if(typeof opts[param.name] !== 'undefined')
        headers[key] = headerValue;
    }
    else if(param.in === 'body' && typeof args[param.name] !== 'undefined') {
      body = args[param.name];
    }
  }

  // if there's a body, need to set the consumes header via requestContentType
  if (body && (this.method === 'post' || this.method === 'put' || this.method === 'patch' || this.method === 'delete')) {
    if (opts.requestContentType)
      consumes = opts.requestContentType;
  } else {
    // if any form params, content type must be set
    if(definedFormParams.length > 0) {
      if(opts.requestContentType)           // override if set
        consumes = opts.requestContentType;
      else if(definedFileParams.length > 0) // if a file, must be multipart/form-data
        consumes = 'multipart/form-data';
      else                                  // default to x-www-from-urlencoded
        consumes = 'application/x-www-form-urlencoded';
    }
    else if (this.type == 'DELETE')
      body = '{}';
    else if (this.type != 'DELETE')
      consumes = null;
  }

  if (consumes && this.consumes) {
    if (this.consumes.indexOf(consumes) === -1) {
      log('server doesn\'t consume ' + consumes + ', try ' + JSON.stringify(this.consumes));
    }
  }

  if (opts.responseContentType) {
    accepts = opts.responseContentType;
  } else {
    accepts = 'application/json';
  }
  if (accepts && this.produces) {
    if (this.produces.indexOf(accepts) === -1) {
      log('server can\'t produce ' + accepts);
    }
  }

  if ((consumes && body !== '') || (consumes === 'application/x-www-form-urlencoded'))
    headers['Content-Type'] = consumes;
  if (accepts)
    headers.Accept = accepts;
  return headers;
};

Operation.prototype.asCurl = function (args) {
  var results = [];
  var headers = this.getHeaderParams(args);
  if (headers) {
    var key;
    for (key in headers)
      results.push("--header \"" + key + ": " + headers[key] + "\"");
  }
  return "curl " + (results.join(" ")) + " " + this.urlify(args);
};

Operation.prototype.encodePathCollection = function(type, name, value) {
  var encoded = '';
  var i;
  var separator = '';
  if(type === 'ssv')
    separator = '%20';
  else if(type === 'tsv')
    separator = '\\t';
  else if(type === 'pipes')
    separator = '|';
  else
    separator = ',';

  for(i = 0; i < value.length; i++) {
    if(i === 0)
      encoded = this.encodeQueryParam(value[i]);
    else
      encoded += separator + this.encodeQueryParam(value[i]);
  }
  return encoded;
};

Operation.prototype.encodeQueryCollection = function(type, name, value) {
  var encoded = '';
  var i;
  if(type === 'default' || type === 'multi') {
    for(i = 0; i < value.length; i++) {
      if(i > 0) encoded += '&';
      encoded += this.encodeQueryParam(name) + '=' + this.encodeQueryParam(value[i]);
    }
  }
  else {
    var separator = '';
    if(type === 'csv')
      separator = ',';
    else if(type === 'ssv')
      separator = '%20';
    else if(type === 'tsv')
      separator = '\\t';
    else if(type === 'pipes')
      separator = '|';
    else if(type === 'brackets') {
      for(i = 0; i < value.length; i++) {
        if(i !== 0)
          encoded += '&';
        encoded += this.encodeQueryParam(name) + '[]=' + this.encodeQueryParam(value[i]);
      }
    }
    if(separator !== '') {
      for(i = 0; i < value.length; i++) {
        if(i === 0)
          encoded = this.encodeQueryParam(name) + '=' + this.encodeQueryParam(value[i]);
        else
          encoded += separator + this.encodeQueryParam(value[i]);
      }
    }
  }
  return encoded;
};

Operation.prototype.encodeQueryParam = function(arg) {
  return encodeURIComponent(arg);
};

/**
 * TODO revisit, might not want to leave '/'
 **/
Operation.prototype.encodePathParam = function(pathParam) {
  var encParts, part, parts, i, len;
  pathParam = pathParam.toString();
  if (pathParam.indexOf('/') === -1) {
    return encodeURIComponent(pathParam);
  } else {
    parts = pathParam.split('/');
    encParts = [];
    for (i = 0, len = parts.length; i < len; i++) {
      encParts.push(encodeURIComponent(parts[i]));
    }
    return encParts.join('/');
  }
};

var Model = function(name, definition) {
  this.name = name;
  this.definition = definition || {};
  this.properties = [];
  var requiredFields = definition.required || [];
  if(definition.type === 'array') {
    var out = new ArrayModel(definition);
    return out;
  }
  var key;
  var props = definition.properties;
  if(props) {
    for(key in props) {
      var required = false;
      var property = props[key];
      if(requiredFields.indexOf(key) >= 0)
        required = true;
      this.properties.push(new Property(key, property, required));
    }
  }
};

Model.prototype.createJSONSample = function(modelsToIgnore) {
  var i, result = {};
  modelsToIgnore = (modelsToIgnore||{});
  modelsToIgnore[this.name] = this;
  for (i = 0; i < this.properties.length; i++) {
    prop = this.properties[i];
    var sample = prop.getSampleValue(modelsToIgnore);
    result[prop.name] = sample;
  }
  delete modelsToIgnore[this.name];
  return result;
};

Model.prototype.getSampleValue = function(modelsToIgnore) {
  var i, obj = {};
  for(i = 0; i < this.properties.length; i++ ) {
    var property = this.properties[i];
    obj[property.name] = property.sampleValue(false, modelsToIgnore);
  }
  return obj;
};

Model.prototype.getMockSignature = function(modelsToIgnore) {
  var i, prop, propertiesStr = [];
  for (i = 0; i < this.properties.length; i++) {
    prop = this.properties[i];
    propertiesStr.push(prop.toString());
  }
  var strong = '<span class="strong">';
  var stronger = '<span class="stronger">';
  var strongClose = '</span>';
  var classOpen = strong + this.name + ' {' + strongClose;
  var classClose = strong + '}' + strongClose;
  var returnVal = classOpen + '<div>' + propertiesStr.join(',</div><div>') + '</div>' + classClose;
  if (!modelsToIgnore)
    modelsToIgnore = {};

  modelsToIgnore[this.name] = this;
  for (i = 0; i < this.properties.length; i++) {
    prop = this.properties[i];
    var ref = prop.$ref;
    var model = models[ref];
    if (model && typeof modelsToIgnore[model.name] === 'undefined') {
      returnVal = returnVal + ('<br>' + model.getMockSignature(modelsToIgnore));
    }
  }
  return returnVal;
};

var Property = function(name, obj, required) {
  this.schema = obj;
  this.required = required;
  if(obj.$ref)
    this.$ref = simpleRef(obj.$ref);
  else if (obj.type === 'array' && obj.items) {
    if(obj.items.$ref)
      this.$ref = simpleRef(obj.items.$ref);
    else
      obj = obj.items;
  }
  this.name = name;
  this.description = obj.description;
  this.obj = obj;
  this.optional = true;
  this.optional = !required;
  this.default = obj.default || null;
  this.example = obj.example || null;
  this.collectionFormat = obj.collectionFormat || null;
  this.maximum = obj.maximum || null;
  this.exclusiveMaximum = obj.exclusiveMaximum || null;
  this.minimum = obj.minimum || null;
  this.exclusiveMinimum = obj.exclusiveMinimum || null;
  this.maxLength = obj.maxLength || null;
  this.minLength = obj.minLength || null;
  this.pattern = obj.pattern || null;
  this.maxItems = obj.maxItems || null;
  this.minItems = obj.minItems || null;
  this.uniqueItems = obj.uniqueItems || null;
  this['enum'] = obj['enum'] || null;
  this.multipleOf = obj.multipleOf || null;
};

Property.prototype.getSampleValue = function (modelsToIgnore) {
  return this.sampleValue(false, modelsToIgnore);
};

Property.prototype.isArray = function () {
  var schema = this.schema;
  if(schema.type === 'array')
    return true;
  else
    return false;
};

Property.prototype.sampleValue = function(isArray, ignoredModels) {
  isArray = (isArray || this.isArray());
  ignoredModels = (ignoredModels || {});
  var type = getStringSignature(this.obj, true);
  var output;

  if(this.$ref) {
    var refModelName = simpleRef(this.$ref);
    var refModel = models[refModelName];
    if(refModel && typeof ignoredModels[type] === 'undefined') {
      ignoredModels[type] = this;
      output = refModel.getSampleValue(ignoredModels);
    }
    else {
      output = refModelName;
    }
  }
  else if(this.example)
    output = this.example;
  else if(this.default)
    output = this.default;
  else if(type === 'date-time')
    output = new Date().toISOString();
  else if(type === 'date')
    output = new Date().toISOString().split("T")[0];
  else if(type === 'string')
    output = 'string';
  else if(type === 'integer')
    output = 0;
  else if(type === 'long')
    output = 0;
  else if(type === 'float')
    output = 0.0;
  else if(type === 'double')
    output = 0.0;
  else if(type === 'boolean')
    output = true;
  else
    output = {};
  ignoredModels[type] = output;
  if(isArray)
    return [output];
  else
    return output;
};

getStringSignature = function(obj, baseComponent) {
  var str = '';
  if(typeof obj.$ref !== 'undefined')
    str += simpleRef(obj.$ref);
  else if(typeof obj.type === 'undefined')
    str += 'object';
  else if(obj.type === 'array') {
    if(baseComponent)
      str += getStringSignature((obj.items || obj.$ref || {}));
    else {
      str += 'Array[';
      str += getStringSignature((obj.items || obj.$ref || {}));
      str += ']';
    }
  }
  else if(obj.type === 'integer' && obj.format === 'int32')
    str += 'integer';
  else if(obj.type === 'integer' && obj.format === 'int64')
    str += 'long';
  else if(obj.type === 'integer' && typeof obj.format === 'undefined')
    str += 'long';
  else if(obj.type === 'string' && obj.format === 'date-time')
    str += 'date-time';
  else if(obj.type === 'string' && obj.format === 'date')
    str += 'date';
  else if(obj.type === 'string' && typeof obj.format === 'undefined')
    str += 'string';
  else if(obj.type === 'number' && obj.format === 'float')
    str += 'float';
  else if(obj.type === 'number' && obj.format === 'double')
    str += 'double';
  else if(obj.type === 'number' && typeof obj.format === 'undefined')
    str += 'double';
  else if(obj.type === 'boolean')
    str += 'boolean';
  else if(obj.$ref)
    str += simpleRef(obj.$ref);
  else
    str += obj.type;
  return str;
};

simpleRef = function(name) {
  if(typeof name === 'undefined')
    return null;
  if(name.indexOf("#/definitions/") === 0)
    return name.substring('#/definitions/'.length);
  else
    return name;
};

Property.prototype.toString = function() {
  var str = getStringSignature(this.obj);
  if(str !== '') {
    str = '<span class="propName ' + this.required + '">' + this.name + '</span> (<span class="propType">' + str + '</span>';
    if(!this.required)
      str += ', <span class="propOptKey">optional</span>';
    str += ')';
  }
  else
    str = this.name + ' (' + JSON.stringify(this.obj) + ')';

  if(typeof this.description !== 'undefined')
    str += ': ' + this.description;

  if (this['enum']) {
    str += ' = <span class="propVals">[\'' + this['enum'].join('\' or \'') + '\']</span>';
  }
  if (this.descr) {
    str += ': <span class="propDesc">' + this.descr + '</span>';
  }


  var options = ''; 
  var isArray = this.schema.type === 'array';
  var type;

  if(isArray) {
    if(this.schema.items)
      type = this.schema.items.type;
    else
      type = '';
  }
  else {
    this.schema.type;
  }

  if (this.default)
    options += optionHtml('Default', this.default);

  switch (type) {
    case 'string':
      if (this.minLength)
        options += optionHtml('Min. Length', this.minLength);
      if (this.maxLength)
        options += optionHtml('Max. Length', this.maxLength);
      if (this.pattern)
        options += optionHtml('Reg. Exp.', this.pattern);
      break;
    case 'integer':
    case 'number':
      if (this.minimum)
        options += optionHtml('Min. Value', this.minimum);
      if (this.exclusiveMinimum)
        options += optionHtml('Exclusive Min.', "true");
      if (this.maximum)
        options += optionHtml('Max. Value', this.maximum);
      if (this.exclusiveMaximum)
        options += optionHtml('Exclusive Max.', "true");
      if (this.multipleOf)
        options += optionHtml('Multiple Of', this.multipleOf);
      break;
  }

  if (isArray) {
    if (this.minItems)
      options += optionHtml('Min. Items', this.minItems);
    if (this.maxItems)
      options += optionHtml('Max. Items', this.maxItems);
    if (this.uniqueItems)
      options += optionHtml('Unique Items', "true");
    if (this.collectionFormat)
      options += optionHtml('Coll. Format', this.collectionFormat);
  }

  if (this['enum']) {
    var enumString;

    if (type === 'number' || type === 'integer')
      enumString = this['enum'].join(', ');
    else {
      enumString = '"' + this['enum'].join('", "') + '"';
    }

    options += optionHtml('Enum', enumString);
  }     

  if (options.length > 0)
    str = '<span class="propWrap">' + str + '<table class="optionsWrapper"><tr><th colspan="2">' + this.name + '</th></tr>' + options + '</table></span>';
  
  return str;
};

optionHtml = function(label, value) {
  return '<tr><td class="optionName">' + label + ':</td><td>' + value + '</td></tr>';
}

typeFromJsonSchema = function(type, format) {
  var str;
  if(type === 'integer' && format === 'int32')
    str = 'integer';
  else if(type === 'integer' && format === 'int64')
    str = 'long';
  else if(type === 'integer' && typeof format === 'undefined')
    str = 'long';
  else if(type === 'string' && format === 'date-time')
    str = 'date-time';
  else if(type === 'string' && format === 'date')
    str = 'date';
  else if(type === 'number' && format === 'float')
    str = 'float';
  else if(type === 'number' && format === 'double')
    str = 'double';
  else if(type === 'number' && typeof format === 'undefined')
    str = 'double';
  else if(type === 'boolean')
    str = 'boolean';
  else if(type === 'string')
    str = 'string';

  return str;
};

var sampleModels = {};
var cookies = {};
var models = {};

SwaggerClient.prototype.buildFrom1_2Spec = function (response) {
  if (response.apiVersion != null) {
    this.apiVersion = response.apiVersion;
  }
  this.apis = {};
  this.apisArray = [];
  this.consumes = response.consumes;
  this.produces = response.produces;
  this.authSchemes = response.authorizations;
  this.info = this.convertInfo(response.info);

  var isApi = false, i, res;
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
    res = new SwaggerResource(response, this);
    this.apis[newName] = res;
    this.apisArray.push(res);
  } else {
    var k;
    for (k = 0; k < response.apis.length; k++) {
      var resource = response.apis[k];
      res = new SwaggerResource(resource, this);
      this.apis[res.name] = res;
      this.apisArray.push(res);
    }
  }
  this.isValid = true;
  if (typeof this.success === 'function') {
    this.success();
  }
  return this;
};

SwaggerClient.prototype.buildFrom1_1Spec = function (response) {
  log('This API is using a deprecated version of Swagger!  Please see http://github.com/wordnik/swagger-core/wiki for more info');
  if (response.apiVersion != null)
    this.apiVersion = response.apiVersion;
  this.apis = {};
  this.apisArray = [];
  this.produces = response.produces;
  this.info = this.convertInfo(response.info);
  var isApi = false, res;
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
    res = new SwaggerResource(response, this);
    this.apis[newName] = res;
    this.apisArray.push(res);
  } else {
    for (k = 0; k < response.apis.length; k++) {
      resource = response.apis[k];
      res = new SwaggerResource(resource, this);
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

SwaggerClient.prototype.convertInfo = function (resp) {
  if(typeof resp == 'object') {
    var info = {}

    info.title = resp.title;
    info.description = resp.description;
    info.termsOfService = resp.termsOfServiceUrl;
    info.contact = {};
    info.contact.name = resp.contact;
    info.license = {};
    info.license.name = resp.license;
    info.license.url = resp.licenseUrl;

    return info;
  }
};

SwaggerClient.prototype.selfReflect = function () {
  var resource, resource_name, ref;
  if (this.apis === null) {
    return false;
  }
  ref = this.apis;
  for (resource_name in ref) {
    resource = ref[resource_name];
    if (resource.ready === null) {
      return false;
    }
  }
  this.setConsolidatedModels();
  this.ready = true;
  if (typeof this.success === 'function') {
    return this.success();
  }
};

SwaggerClient.prototype.setConsolidatedModels = function () {
  var model, modelName, resource, resource_name, i, apis, models, results;
  this.models = {};
  apis = this.apis;
  for (resource_name in apis) {
    resource = apis[resource_name];
    for (modelName in resource.models) {
      if (typeof this.models[modelName] === 'undefined') {
        this.models[modelName] = resource.models[modelName];
        this.modelsArray.push(resource.models[modelName]);
      }
    }
  }
  models = this.modelsArray;
  results = [];
  for (i = 0; i < models.length; i++) {
    model = models[i];
    results.push(model.setReferencedModels(this.models));
  }
  return results;
};

var SwaggerResource = function (resourceObj, api) {
  var _this = this;
  this.api = api;
  this.swaggerRequstHeaders = api.swaggerRequstHeaders;
  this.path = (typeof this.api.resourcePath === 'string') ? this.api.resourcePath : resourceObj.path;
  this.description = resourceObj.description;
  this.authorizations = (resourceObj.authorizations || {});

  var parts = this.path.split('/');
  this.name = parts[parts.length - 1].replace('.{format}', '');
  this.basePath = this.api.basePath;
  this.operations = {};
  this.operationsArray = [];
  this.modelsArray = [];
  this.models = {};
  this.rawModels = {};
  this.useJQuery = (typeof api.useJQuery !== 'undefined') ? api.useJQuery : null;

  if ((resourceObj.apis) && this.api.resourcePath) {
    this.addApiDeclaration(resourceObj);
  } else {
    if (typeof this.path === 'undefined') {
      this.api.fail('SwaggerResources must have a path.');
    }
    if (this.path.substring(0, 4) === 'http') {
      this.url = this.path.replace('{format}', 'json');
    } else {
      this.url = this.api.basePath + this.path.replace('{format}', 'json');
    }
    this.api.progress('fetching resource ' + this.name + ': ' + this.url);
    var obj = {
      url: this.url,
      method: 'GET',
      useJQuery: this.useJQuery,
      headers: {
        accept: this.swaggerRequstHeaders
      },
      on: {
        response: function (resp) {
          var responseObj = resp.obj || JSON.parse(resp.data);
          return _this.addApiDeclaration(responseObj);
        },
        error: function (response) {
          return _this.api.fail('Unable to read api \'' +
          _this.name + '\' from path ' + _this.url + ' (server returned ' + response.statusText + ')');
        }
      }
    };
    var e = typeof window !== 'undefined' ? window : exports;
    e.authorizations.apply(obj);
    new SwaggerHttp().execute(obj);
  }
};

SwaggerResource.prototype.getAbsoluteBasePath = function (relativeBasePath) {
  var pos, url;
  url = this.api.basePath;
  pos = url.lastIndexOf(relativeBasePath);
  var parts = url.split('/');
  var rootUrl = parts[0] + '//' + parts[2];

  if (relativeBasePath.indexOf('http') === 0)
    return relativeBasePath;
  if (relativeBasePath === '/')
    return rootUrl;
  if (relativeBasePath.substring(0, 1) == '/') {
    // use root + relative
    return rootUrl + relativeBasePath;
  }
  else {
    pos = this.basePath.lastIndexOf('/');
    var base = this.basePath.substring(0, pos);
    if (base.substring(base.length - 1) == '/')
      return base + relativeBasePath;
    else
      return base + '/' + relativeBasePath;
  }
};

SwaggerResource.prototype.addApiDeclaration = function (response) {
  if (typeof response.produces === 'string')
    this.produces = response.produces;
  if (typeof response.consumes === 'string')
    this.consumes = response.consumes;
  if ((typeof response.basePath === 'string') && response.basePath.replace(/\s/g, '').length > 0)
    this.basePath = response.basePath.indexOf('http') === -1 ? this.getAbsoluteBasePath(response.basePath) : response.basePath;

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
  if (typeof models === 'object') {
    var modelName;
    for (modelName in models) {
      if (typeof this.models[modelName] === 'undefined') {
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
      if (typeof o.consumes !== 'undefined')
        consumes = o.consumes;
      else
        consumes = this.consumes;

      if (typeof o.produces !== 'undefined')
        produces = o.produces;
      else
        produces = this.produces;
      var type = (o.type || o.responseClass);

      if (type === 'array') {
        ref = null;
        if (o.items)
          ref = o.items.type || o.items.$ref;
        type = 'array[' + ref + ']';
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
      var op = new SwaggerOperation(o.nickname,
          resource_path,
          method,
          o.parameters,
          o.summary,
          o.notes,
          type,
          responseMessages, 
          this, 
          consumes, 
          produces, 
          o.authorizations, 
          o.deprecated);

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

var SwaggerModel = function (modelName, obj) {
  this.name = typeof obj.id !== 'undefined' ? obj.id : modelName;
  this.properties = [];
  var propertyName;
  for (propertyName in obj.properties) {
    if (obj.required) {
      var value;
      for (value in obj.required) {
        if (propertyName === obj.required[value]) {
          obj.properties[propertyName].required = true;
        }
      }
    }
    var prop = new SwaggerModelProperty(propertyName, obj.properties[propertyName], this);
    this.properties.push(prop);
  }
};

SwaggerModel.prototype.setReferencedModels = function (allModels) {
  var results = [];
  for (var i = 0; i < this.properties.length; i++) {
    var property = this.properties[i];
    var type = property.type || property.dataType;
    if (allModels[type])
      results.push(property.refModel = allModels[type]);
    else if ((property.refDataType) && (allModels[property.refDataType]))
      results.push(property.refModel = allModels[property.refDataType]);
    else
      results.push(void 0);
  }
  return results;
};

SwaggerModel.prototype.getMockSignature = function (modelsToIgnore) {
  var i, prop, propertiesStr = [];
  for (i = 0; i < this.properties.length; i++) {
    prop = this.properties[i];
    propertiesStr.push(prop.toString());
  }

  var strong = '<span class="strong">';
  var strongClose = '</span>';
  var classOpen = strong + this.name + ' {' + strongClose;
  var classClose = strong + '}' + strongClose;
  var returnVal = classOpen + '<div>' + propertiesStr.join(',</div><div>') + '</div>' + classClose;
  if (!modelsToIgnore)
    modelsToIgnore = [];
  modelsToIgnore.push(this.name);

  for (i = 0; i < this.properties.length; i++) {
    prop = this.properties[i];
    if ((prop.refModel) && modelsToIgnore.indexOf(prop.refModel.name) === -1) {
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
    modelsToIgnore = (modelsToIgnore || []);
    modelsToIgnore.push(this.name);
    for (var i = 0; i < this.properties.length; i++) {
      var prop = this.properties[i];
      result[prop.name] = prop.getSampleValue(modelsToIgnore);
    }
    modelsToIgnore.pop(this.name);
    return result;
  }
};

var SwaggerModelProperty = function (name, obj, model) {
  this.name = name;
  this.dataType = obj.type || obj.dataType || obj.$ref;
  this.isCollection = this.dataType && (this.dataType.toLowerCase() === 'array' || this.dataType.toLowerCase() === 'list' || this.dataType.toLowerCase() === 'set');
  this.descr = obj.description;
  this.required = obj.required;
  this.defaultValue = applyModelPropertyMacro(obj, model);
  if (obj.items) {
    if (obj.items.type) {
      this.refDataType = obj.items.type;
    }
    if (obj.items.$ref) {
      this.refDataType = obj.items.$ref;
    }
  }
  this.dataTypeWithRef = this.refDataType ? (this.dataType + '[' + this.refDataType + ']') : this.dataType;
  if (obj.allowableValues) {
    this.valueType = obj.allowableValues.valueType;
    this.values = obj.allowableValues.values;
    if (this.values) {
      this.valuesString = '\'' + this.values.join('\' or \'') + '\'';
    }
  }
  if (obj['enum']) {
    this.valueType = 'string';
    this.values = obj['enum'];
    if (this.values) {
      this.valueString = '\'' + this.values.join('\' or \'') + '\'';
    }
  }
};

SwaggerModelProperty.prototype.getSampleValue = function (modelsToIgnore) {
  var result;
  if ((this.refModel) && (modelsToIgnore.indexOf(this.refModel.name) === -1)) {
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
  if ((typeof this.defaultValue !== 'undefined') && this.defaultValue) {
    result = this.defaultValue;
  } else if (value === 'integer') {
    result = 0;
  } else if (value === 'boolean') {
    result = false;
  } else if (value === 'double' || value === 'number') {
    result = 0.0;
  } else if (value === 'string') {
    result = '';
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
  if (this.values) {
    str += ' = <span class="propVals">[\'' + this.values.join('\' or \'') + '\']</span>';
  }
  if (this.descr) {
    str += ': <span class="propDesc">' + this.descr + '</span>';
  }
  return str;
};

var SwaggerOperation = function (nickname, path, method, parameters, summary, notes, type, responseMessages, resource, consumes, produces, authorizations, deprecated) {
  var _this = this;

  var errors = [];
  this.nickname = (nickname || errors.push('SwaggerOperations must have a nickname.'));
  this.path = (path || errors.push('SwaggerOperation ' + nickname + ' is missing path.'));
  this.method = (method || errors.push('SwaggerOperation ' + nickname + ' is missing method.'));
  this.parameters = parameters ? parameters : [];
  this.summary = summary;
  this.notes = notes;
  this.type = type;
  this.responseMessages = (responseMessages || []);
  this.resource = (resource || errors.push('Resource is required'));
  this.consumes = consumes;
  this.produces = produces;
  this.authorizations = typeof authorizations !== 'undefined' ? authorizations : resource.authorizations;
  this.deprecated = (typeof deprecated === 'string' ? Boolean(deprecated) : deprecated);
  this['do'] = __bind(this['do'], this);

  if (errors.length > 0) {
    console.error('SwaggerOperation errors', errors, arguments);
    this.resource.api.fail(errors);
  }

  this.path = this.path.replace('{format}', 'json');
  this.method = this.method.toLowerCase();
  this.isGetMethod = this.method === 'GET';

  var i, j, v;
  this.resourceName = this.resource.name;
  if (typeof this.type !== 'undefined' && this.type === 'void')
    this.type = null;
  else {
    this.responseClassSignature = this.getSignature(this.type, this.resource.models);
    this.responseSampleJSON = this.getSampleJSON(this.type, this.resource.models);
  }

  for (i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    // might take this away
    param.name = param.name || param.type || param.dataType;
    // for 1.1 compatibility
    type = param.type || param.dataType;
    if (type === 'array') {
      type = 'array[' + (param.items.$ref ? param.items.$ref : param.items.type) + ']';
    }
    param.type = type;

    if (type && type.toLowerCase() === 'boolean') {
      param.allowableValues = {};
      param.allowableValues.values = ['true', 'false'];
    }
    param.signature = this.getSignature(type, this.resource.models);
    param.sampleJSON = this.getSampleJSON(type, this.resource.models);

    var enumValue = param['enum'];
    if (typeof enumValue !== 'undefined') {
      param.isList = true;
      param.allowableValues = {};
      param.allowableValues.descriptiveValues = [];

      for (j = 0; j < enumValue.length; j++) {
        v = enumValue[j];
        if (param.defaultValue) {
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
      if (param.allowableValues.valueType === 'RANGE')
        param.isRange = true;
      else
        param.isList = true;
      if (param.allowableValues != null) {
        param.allowableValues.descriptiveValues = [];
        if (param.allowableValues.values) {
          for (j = 0; j < param.allowableValues.values.length; j++) {
            v = param.allowableValues.values[j];
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
    param.defaultValue = applyParameterMacro(this, param);
  }
  var defaultSuccessCallback = this.resource.api.defaultSuccessCallback || null;
  var defaultErrorCallback = this.resource.api.defaultErrorCallback || null;

  this.resource[this.nickname] = function (args, opts, callback, error) {
    var arg1, arg2, arg3, arg4;
    if(typeof args === 'function') {  // right shift 3
      arg1 = {}; arg2 = {}; arg3 = args; arg4 = opts;
    }
    else if(typeof args === 'object' && typeof opts === 'function') { // right shift 2
      arg1 = args; arg2 = {}; arg3 = opts; arg4 = callback;
    }
    else {
      arg1 = args; arg2 = opts; arg3 = callback; arg4 = error;
    }
    return _this['do'](arg1 || {}, arg2 || {}, arg3 || defaultSuccessCallback, arg4 || defaultErrorCallback);
  };

  this.resource[this.nickname].help = function () {
    return _this.help();
  };
  this.resource[this.nickname].asCurl = function (args) {
    return _this.asCurl(args);
  };
};

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
  isPrimitive = ((typeof listType !== 'undefined') && models[listType]) || (typeof models[type] !== 'undefined') ? false : true;
  if (isPrimitive) {
    return type;
  } else {
    if (typeof listType !== 'undefined') {
      return models[listType].getMockSignature();
    } else {
      return models[type].getMockSignature();
    }
  }
};

SwaggerOperation.prototype.getSampleJSON = function (type, models) {
  var isPrimitive, listType, val;
  listType = this.isListType(type);
  isPrimitive = ((typeof listType !== 'undefined') && models[listType]) || (typeof models[type] !== 'undefined') ? false : true;
  val = isPrimitive ? void 0 : (listType != null ? models[listType].createJSONSample() : models[type].createJSONSample());
  if (val) {
    val = listType ? [val] : val;
    if (typeof val == 'string')
      return val;
    else if (typeof val === 'object') {
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

SwaggerOperation.prototype['do'] = function (args, opts, callback, error) {
  var key, param, params, possibleParams = [], req, value;

  if (typeof error !== 'function') {
    error = function (xhr, textStatus, error) {
      return log(xhr, textStatus, error);
    };
  }

  if (typeof callback !== 'function') {
    callback = function (response) {
      var content;
      content = null;
      if (response != null) {
        content = response.data;
      } else {
        content = 'no data';
      }
      return log('default callback: ' + content);
    };
  }

  params = {};
  params.headers = [];
  if (args.headers != null) {
    params.headers = args.headers;
    delete args.headers;
  }
  // allow override from the opts
  if(opts && opts.responseContentType) {
    params.headers['Content-Type'] = opts.responseContentType;
  }
  if(opts && opts.requestContentType) {
    params.headers.Accept = opts.requestContentType;
  }

  for (var i = 0; i < this.parameters.length; i++) {
    param = this.parameters[i];
    if (param.paramType === 'header') {
      if (typeof args[param.name] !== 'undefined')
        params.headers[param.name] = args[param.name];
    }
    else if (param.paramType === 'form' || param.paramType.toLowerCase() === 'file')
      possibleParams.push(param);
    else if (param.paramType === 'body' && param.name !== 'body' && typeof args[param.name] !== 'undefined') {
      if (args.body) {
        throw new Error('Saw two body params in an API listing; expecting a max of one.');
      }
      args.body = args[param.name];
    }
  }

  if (typeof args.body !== 'undefined') {
    params.body = args.body;
    delete args.body;
  }

  if (possibleParams) {
    for (key in possibleParams) {
      value = possibleParams[key];
      if (args[value.name]) {
        params[value.name] = args[value.name];
      }
    }
  }

  req = new SwaggerRequest(this.method, this.urlify(args), params, opts, callback, error, this);
  if (opts.mock) {
    return req;
  } else {
    return true;
  }
};

SwaggerOperation.prototype.pathJson = function () {
  return this.path.replace('{format}', 'json');
};

SwaggerOperation.prototype.pathXml = function () {
  return this.path.replace('{format}', 'xml');
};

SwaggerOperation.prototype.encodePathParam = function (pathParam) {
  var encParts, part, parts, _i, _len;
  pathParam = pathParam.toString();
  if (pathParam.indexOf('/') === -1) {
    return encodeURIComponent(pathParam);
  } else {
    parts = pathParam.split('/');
    encParts = [];
    for (_i = 0, _len = parts.length; _i < _len; _i++) {
      part = parts[_i];
      encParts.push(encodeURIComponent(part));
    }
    return encParts.join('/');
  }
};

SwaggerOperation.prototype.urlify = function (args) {
  var i, j, param, url;
  // ensure no double slashing...
  if(this.resource.basePath.length > 1 && this.resource.basePath.slice(-1) === '/' && this.pathJson().charAt(0) === '/')
    url = this.resource.basePath + this.pathJson().substring(1);
  else
    url = this.resource.basePath + this.pathJson();
  var params = this.parameters;
  for (i = 0; i < params.length; i++) {
    param = params[i];
    if (param.paramType === 'path') {
      if (typeof args[param.name] !== 'undefined') {
        // apply path params and remove from args
        var reg = new RegExp('\\{\\s*?' + param.name + '.*?\\}(?=\\s*?(\\/?|$))', 'gi');
        url = url.replace(reg, this.encodePathParam(args[param.name]));
        delete args[param.name];
      }
      else
        throw '' + param.name + ' is a required path param.';
    }
  }

  var queryParams = '';
  for (i = 0; i < params.length; i++) {
    param = params[i];
    if(param.paramType === 'query') {
      if (queryParams !== '')
        queryParams += '&';    
      if (Array.isArray(param)) {
        var output = '';   
        for(j = 0; j < param.length; j++) {    
          if(j > 0)    
            output += ',';   
          output += encodeURIComponent(param[j]);    
        }    
        queryParams += encodeURIComponent(param.name) + '=' + output;    
      }
      else {
        if (typeof args[param.name] !== 'undefined') {
          queryParams += encodeURIComponent(param.name) + '=' + encodeURIComponent(args[param.name]);
        } else {
          if (param.required)
            throw '' + param.name + ' is a required query param.';
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
  var msg = '';
  var params = this.parameters;
  for (var i = 0; i < params.length; i++) {
    var param = params[i];
    if (msg !== '')
      msg += '\n';
    msg += '* ' + param.name + (param.required ? ' (required)' : '') + " - " + param.description;
  }
  return msg;
};

SwaggerOperation.prototype.asCurl = function (args) {
  var results = [];
  var i;

  var headers = SwaggerRequest.prototype.setHeaders(args, {}, this);    
  for(i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    if(param.paramType && param.paramType === 'header' && args[param.name]) {
      headers[param.name] = args[param.name];
    }
  }

  var key;
  for (key in headers) {
    results.push('--header "' + key + ': ' + headers[key] + '"');
  }
  return 'curl ' + (results.join(' ')) + ' ' + this.urlify(args);
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
      formatted = formatted.substr(0, formatted.length - 1) + ln + '\n';
    } else {
      formatted += padding + ln + '\n';
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
  this.type = (type || errors.push('SwaggerRequest type is required (get/post/put/delete/patch/options).'));
  this.url = (url || errors.push('SwaggerRequest url is required.'));
  this.params = params;
  this.opts = opts;
  this.successCallback = (successCallback || errors.push('SwaggerRequest successCallback is required.'));
  this.errorCallback = (errorCallback || errors.push('SwaggerRequest error callback is required.'));
  this.operation = (operation || errors.push('SwaggerRequest operation is required.'));
  this.execution = execution;
  this.headers = (params.headers || {});

  if (errors.length > 0) {
    throw errors;
  }

  this.type = this.type.toUpperCase();

  // set request, response content type headers
  var headers = this.setHeaders(params, opts, this.operation);
  var body = params.body;

  // encode the body for form submits
  if (headers['Content-Type']) {
    var key, value, values = {}, i;
    var operationParams = this.operation.parameters;
    for (i = 0; i < operationParams.length; i++) {
      var param = operationParams[i];
      if (param.paramType === 'form')
        values[param.name] = param;
    }

    if (headers['Content-Type'].indexOf('application/x-www-form-urlencoded') === 0) {
      var encoded = '';
      for (key in values) {
        value = this.params[key];
        if (typeof value !== 'undefined') {
          if (encoded !== '')
            encoded += '&';
          encoded += encodeURIComponent(key) + '=' + encodeURIComponent(value);
        }
      }
      body = encoded;
    }
    else if (headers['Content-Type'].indexOf('multipart/form-data') === 0) {
      // encode the body for form submits
      var data = '';
      var boundary = '----SwaggerFormBoundary' + Date.now();
      for (key in values) {
        value = this.params[key];
        if (typeof value !== 'undefined') {
          data += '--' + boundary + '\n';
          data += 'Content-Disposition: form-data; name="' + key + '"';
          data += '\n\n';
          data += value + '\n';
        }
      }
      data += '--' + boundary + '--\n';
      headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary;
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

    var status = false;
    if (this.operation.resource && this.operation.resource.api && this.operation.resource.api.clientAuthorizations) {
      // Get the client authorizations from the resource declaration
      status = this.operation.resource.api.clientAuthorizations.apply(obj, this.operation.authorizations);
    } else {
      // Get the client authorization from the default authorization declaration
      var e;
      if (typeof window !== 'undefined') {
        e = window;
      } else {
        e = exports;
      }
      status = e.authorizations.apply(obj, this.operation.authorizations);
    }

    if (!opts.mock) {
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

SwaggerRequest.prototype.setHeaders = function (params, opts, operation) {
  // default type
  var accepts = opts.responseContentType || 'application/json';
  var consumes = opts.requestContentType || 'application/json';

  var allDefinedParams = operation.parameters;
  var definedFormParams = [];
  var definedFileParams = [];
  var body = params.body;
  var headers = {};

  // get params from the operation and set them in definedFileParams, definedFormParams, headers
  var i;
  for (i = 0; i < allDefinedParams.length; i++) {
    var param = allDefinedParams[i];
    if (param.paramType === 'form')
      definedFormParams.push(param);
    else if (param.paramType === 'file')
      definedFileParams.push(param);
    else if (param.paramType === 'header' && this.params.headers) {
      var key = param.name;
      var headerValue = this.params.headers[param.name];
      if (typeof this.params.headers[param.name] !== 'undefined')
        headers[key] = headerValue;
    }
  }

  // if there's a body, need to set the accepts header via requestContentType
  if (body && (this.type === 'POST' || this.type === 'PUT' || this.type === 'PATCH' || this.type === 'DELETE')) {
    if (this.opts.requestContentType)
      consumes = this.opts.requestContentType;
  } else {
    // if any form params, content type must be set
    if (definedFormParams.length > 0) {
      if (definedFileParams.length > 0)
        consumes = 'multipart/form-data';
      else
        consumes = 'application/x-www-form-urlencoded';
    }
    else if (this.type === 'DELETE')
      body = '{}';
    else if (this.type != 'DELETE')
      consumes = null;
  }

  if (consumes && this.operation.consumes) {
    if (this.operation.consumes.indexOf(consumes) === -1) {
      log('server doesn\'t consume ' + consumes + ', try ' + JSON.stringify(this.operation.consumes));
    }
  }

  if (this.opts && this.opts.responseContentType) {
    accepts = this.opts.responseContentType;
  } else {
    accepts = 'application/json';
  }
  if (accepts && operation.produces) {
    if (operation.produces.indexOf(accepts) === -1) {
      log('server can\'t produce ' + accepts);
    }
  }

  if ((consumes && body !== '') || (consumes === 'application/x-www-form-urlencoded'))
    headers['Content-Type'] = consumes;
  if (accepts)
    headers.Accept = accepts;
  return headers;
};

/**
 * SwaggerHttp is a wrapper for executing requests
 */
var SwaggerHttp = function() {};

SwaggerHttp.prototype.execute = function(obj) {
  if(obj && (typeof obj.useJQuery === 'boolean'))
    this.useJQuery = obj.useJQuery;
  else
    this.useJQuery = this.isIE8();

  if(obj && typeof obj.body === 'object') {
    obj.body = JSON.stringify(obj.body);
  }

  if(this.useJQuery)
    return new JQueryHttpClient().execute(obj);
  else
    return new ShredHttpClient().execute(obj);
};

SwaggerHttp.prototype.isIE8 = function() {
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
 *       Since we are using closures here we need to alias it for internal use.
 */
var JQueryHttpClient = function(options) {
  "use strict";
  if(!jQuery){
    var jQuery = window.jQuery;
  }
};

JQueryHttpClient.prototype.execute = function(obj) {
  var cb = obj.on;
  var request = obj;

  obj.type = obj.method;
  obj.cache = false;

  obj.beforeSend = function(xhr) {
    var key, results;
    if (obj.headers) {
      results = [];
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
  obj.complete = function(response, textStatus, opts) {
    var headers = {},
      headerArray = response.getAllResponseHeaders().split("\n");

    for(var i = 0; i < headerArray.length; i++) {
      var toSplit = headerArray[i].trim();
      if(toSplit.length === 0)
        continue;
      var separator = toSplit.indexOf(":");
      if(separator === -1) {
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

    var contentType = (headers["content-type"]||headers["Content-Type"]||null);
    if(contentType) {
      if(contentType.indexOf("application/json") === 0 || contentType.indexOf("+json") > 0) {
        try {
          out.obj = response.responseJSON || {};
        } catch (ex) {
          // do not set out.obj
          log("unable to parse JSON content");
        }
      }
    }

    if(response.status >= 200 && response.status < 300)
      cb.response(out);
    else if(response.status === 0 || (response.status >= 400 && response.status < 599))
      cb.error(out);
    else
      return cb.response(out);
  };

  jQuery.support.cors = true;
  return jQuery.ajax(obj);
};

/*
 * ShredHttpClient is a light-weight, node or browser HTTP client
 */
var ShredHttpClient = function(options) {
  this.options = (options||{});
  this.isInitialized = false;

  var identity, toString;

  if (typeof window !== 'undefined') {
    this.Shred = require("./shred");
    this.content = require("./shred/content");
  }
  else
    this.Shred = require("shred");
  this.shred = new this.Shred(options);
};

ShredHttpClient.prototype.initShred = function () {
  this.isInitialized = true;
  this.registerProcessors(this.shred);
};

ShredHttpClient.prototype.registerProcessors = function(shred) {
  var identity = function(x) {
    return x;
  };
  var toString = function(x) {
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
};

ShredHttpClient.prototype.execute = function(obj) {
  if(!this.isInitialized)
    this.initShred();

  var cb = obj.on, res;
  var transform = function(response) {
    var out = {
      headers: response._headers,
      url: response.request.url,
      method: response.request.method,
      status: response.status,
      data: response.content.data
    };

    var headers = response._headers.normalized || response._headers;
    var contentType = (headers["content-type"]||headers["Content-Type"]||null);

    if(contentType) {
      if(contentType.indexOf("application/json") === 0 || contentType.indexOf("+json") > 0) {
        if(response.content.data && response.content.data !== "")
          try{
            out.obj = JSON.parse(response.content.data);
          }
          catch (e) {
            // unable to parse
          }
        else
          out.obj = {};
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

  res = {
    error: function (response) {
      if (obj)
        return cb.error(transform(response));
    },
    // Catch the Shred error raised when the request errors as it is made (i.e. No Response is coming)
    request_error: function (err) {
      if (obj)
        return cb.error(transformError(err));
    },
    response: function (response) {
      if (obj) {
        return cb.response(transform(response));
      }
    }
  };
  if (obj) {
    obj.on = res;
  }
  return this.shred.request(obj);
};


var e = (typeof window !== 'undefined' ? window : exports);

e.authorizations = new SwaggerAuthorizations();
e.ApiKeyAuthorization = ApiKeyAuthorization;
e.PasswordAuthorization = PasswordAuthorization;
e.CookieAuthorization = CookieAuthorization;
e.SwaggerClient = SwaggerClient;
e.Operation = Operation;
e.Model = Model;
e.models = models;
})();