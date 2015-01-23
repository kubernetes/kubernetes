// swagger-client.js
// version 2.1.0-alpha.4
/**
 * Array Model
 **/
var ArrayModel = function(definition) {
  this.name = "name";
  this.definition = definition || {};
  this.properties = [];
  this.type;
  this.ref;

  var requiredFields = definition.enum || [];
  var items = definition.items;
  if(items) {
    var type = items.type;
    if(items.type) {
      this.type = typeFromJsonSchema(type.type, type.format);
    }
    else {
      this.ref = items['$ref'];
    }
  }
}

ArrayModel.prototype.createJSONSample = function(modelsToIgnore) {
  var result;
  modelsToIgnore = (modelsToIgnore||{})
  if(this.type) {
    result = type;
  }
  else if (this.ref) {
    var name = simpleRef(this.ref);
    result = models[name].createJSONSample();
  }
  return [ result ];
};

ArrayModel.prototype.getSampleValue = function(modelsToIgnore) {
  var result;
  modelsToIgnore = (modelsToIgnore || {})
  if(this.type) {
    result = type;
  }
  else if (this.ref) {
    var name = simpleRef(this.ref);
    result = models[name].getSampleValue(modelsToIgnore);
  }
  return [ result ];
}

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

SwaggerAuthorizations.prototype.apply = function(obj, authorizations) {
  var status = null;
  var key;

  // if the "authorizations" key is undefined, or has an empty array, add all keys
  if(typeof authorizations === 'undefined' || Object.keys(authorizations).length == 0) {
    for (key in this.authz) {
      value = this.authz[key];
      result = value.apply(obj, authorizations);
      if (result === true)
        status = true;
    }
  }
  else {
    if(Array.isArray(authorizations)) {
      var i;
      for(i = 0; i < authorizations.length; i++) {
        var auth = authorizations[i];
        log(auth);
        for (key in this.authz) {
          var value = this.authz[key];
          if(typeof value !== 'undefined') {
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
}

CookieAuthorization.prototype.apply = function(obj, authorizations) {
  obj.cookieJar = obj.cookieJar || CookieJar();
  obj.cookieJar.setCookie(this.cookie);
  return true;
}

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
  obj.headers["Authorization"] = "Basic " + base64encoder(this.username + ":" + this.password);
  return true;
};var __bind = function(fn, me){
  return function(){
    return fn.apply(me, arguments);
  };
};

fail = function(message) {
  log(message);
}

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
  }
}

if (!('filter' in Array.prototype)) {
  Array.prototype.filter= function(filter, that /*opt*/) {
    var other= [], v;
    for (var i=0, n= this.length; i<n; i++)
      if (i in this && filter.call(that, v= this[i], i, this))
        other.push(v);
    return other;
  };
}

if (!('map' in Array.prototype)) {
  Array.prototype.map= function(mapper, that /*opt*/) {
    var other= new Array(this.length);
    for (var i= 0, n= this.length; i<n; i++)
      if (i in this)
        other[i]= mapper.call(that, this[i], i, this);
    return other;
  };
}

Object.keys = Object.keys || (function () {
  var hasOwnProperty = Object.prototype.hasOwnProperty,
    hasDontEnumBug = !{toString:null}.propertyIsEnumerable("toString"),
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
/**
 * PrimitiveModel
 **/
var PrimitiveModel = function(definition) {
  this.name = "name";
  this.definition = definition || {};
  this.properties = [];
  this.type;

  var requiredFields = definition.enum || [];
  this.type = typeFromJsonSchema(definition.type, definition.format);
}

PrimitiveModel.prototype.createJSONSample = function(modelsToIgnore) {
  var result = this.type;
  return result;
};

PrimitiveModel.prototype.getSampleValue = function() {
  var result = this.type;
  return null;
}

PrimitiveModel.prototype.getMockSignature = function(modelsToIgnore) {
  var propertiesStr = [];
  var i;
  for (i = 0; i < this.properties.length; i++) {
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
    modelsToIgnore = {};
  modelsToIgnore[this.name] = this;
  var i;
  for (i = 0; i < this.properties.length; i++) {
    var prop = this.properties[i];
    var ref = prop['$ref'];
    var model = models[ref];
    if (model && typeof modelsToIgnore[ref] === 'undefined') {
      returnVal = returnVal + ('<br>' + model.getMockSignature(modelsToIgnore));
    }
  }
  return returnVal;
};var SwaggerClient = function(url, options) {
  this.isBuilt = false;
  this.url = null;
  this.debug = false;
  this.basePath = null;
  this.authorizations = null;
  this.authorizationScheme = null;
  this.isValid = false;
  this.info = null;
  this.useJQuery = false;
  this.models = models;

  options = (options||{});
  if (url)
    if (url.url) options = url;
    else this.url = url;
  else options = url;

  if (options.url != null)
    this.url = options.url;

  if (options.success != null)
    this.success = options.success;

  if (typeof options.useJQuery === 'boolean')
    this.useJQuery = options.useJQuery;

  this.failure = options.failure != null ? options.failure : function() {};
  this.progress = options.progress != null ? options.progress : function() {};
  this.spec = options.spec;

  if (options.success != null)
    this.build();
}

SwaggerClient.prototype.build = function() {
  var self = this;
  this.progress('fetching resource list: ' + this.url);
  var obj = {
    useJQuery: this.useJQuery,
    url: this.url,
    method: "get",
    headers: {
      accept: "application/json, */*"
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
          self.isValid = false;
          self.failure();
        }
      }
    }
  };
  if(this.spec) {
    var self = this;
    setTimeout(function() { self.buildFromSpec(self.spec); }, 10);
  }
  else {
    var e = (typeof window !== 'undefined' ? window : exports);
    var status = e.authorizations.apply(obj);
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
  this.scheme;
  this.basePath = response.basePath || '';
  this.apis = {};
  this.apisArray = [];
  this.consumes = response.consumes;
  this.produces = response.produces;
  this.securityDefinitions = response.securityDefinitions;

  // legacy support
  this.authSchemes = response.securityDefinitions;

  var location = this.parseUri(this.url);
  if(typeof this.schemes === 'undefined' || this.schemes.length === 0) {
    this.scheme = location.scheme;
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
              operationGroup.label = tag;
              operationGroup.apis = [];
              this[tag].help = this.help.bind(operationGroup);
              this.apisArray.push(new OperationGroup(tag, operationObject));
            }
            operationGroup[operationId] = operationObject.execute.bind(operationObject);
            operationGroup[operationId].help = operationObject.help.bind(operationObject);
            operationGroup.apis.push(operationObject);

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
}

SwaggerClient.prototype.parseUri = function(uri) {
  var urlParseRE = /^(((([^:\/#\?]+:)?(?:(\/\/)((?:(([^:@\/#\?]+)(?:\:([^:@\/#\?]+))?)@)?(([^:\/#\?\]\[]+|\[[^\/\]@#?]+\])(?:\:([0-9]+))?))?)?)?((\/?(?:[^\/\?#]+\/+)*)([^\?#]*)))?(\?[^#]+)?)(#.*)?/;
  var parts = urlParseRE.exec(uri);
  return {
    scheme: parts[4].replace(':',''),
    host: parts[11],
    port: parts[12],
    path: parts[15]
  };
}

SwaggerClient.prototype.help = function() {
  var i;
  log('operations for the "' + this.label + '" tag');
  for(i = 0; i < this.apis.length; i++) {
    var api = this.apis[i];
    log('  * ' + api.nickname + ': ' + api.operation.summary);
  }
}

SwaggerClient.prototype.tagFromLabel = function(label) {
  return label;
}

SwaggerClient.prototype.idFromOp = function(path, httpMethod, op) {
  if(typeof op.operationId !== 'undefined') {
    return (op.operationId);
  }
  else {
    return path.substring(1)
      .replace(/\//g, "_")
      .replace(/\{/g, "")
      .replace(/\}/g, "")
      .replace(/\./g, "_") + "_" + httpMethod;
  }
}

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
}

var Operation = function(parent, operationId, httpMethod, path, args, definitions) {
  var errors = [];
  this.operation = args;
  this.deprecated = args.deprecated;
  this.consumes = args.consumes;
  this.produces = args.produces;
  this.parent = parent;
  this.host = parent.host;
  this.schemes = parent.schemes;
  this.scheme = parent.scheme || 'http';
  this.basePath = parent.basePath;
  this.nickname = (operationId||errors.push('Operations must have a nickname.'));
  this.method = (httpMethod||errors.push('Operation ' + operationId + ' is missing method.'));
  this.path = (path||errors.push('Operation ' + nickname + ' is missing path.'));
  this.parameters = args != null ? (args.parameters||[]) : {};
  this.summary = args.summary || '';
  this.responses = (args.responses||{});
  this.type = null;
  this.security = args.security;
  this.authorizations = args.security;
  this.description = args.description;

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
      param.enum = ["true", "false"];
    }
    if(typeof param.enum !== 'undefined') {
      var id;
      param.allowableValues = {};
      param.allowableValues.values = [];
      param.allowableValues.descriptiveValues = [];
      for(id = 0; id < param.enum.length; id++) {
        var value = param.enum[id];
        var isDefault = (value === param.default) ? true : false;
        param.allowableValues.values.push(value);
        param.allowableValues.descriptiveValues.push({value : value, isDefault: isDefault});
      }
    }
    if(param.type === 'array' && typeof param.allowableValues === 'undefined') {
      // can't show as a list if no values to select from
      delete param.isList;
      delete param.allowMultiple;
    }
    param.signature = this.getSignature(innerType, models);
    param.sampleJSON = this.getSampleJSON(innerType, models);
    param.responseClassSignature = param.signature;
  }

  var response;
  var model;
  var responses = this.responses;

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
    if(resolvedModel) {
      this.type = resolvedModel.name;
      this.responseSampleJSON = JSON.stringify(resolvedModel.getSampleValue(), null, 2);
      this.responseClassSignature = resolvedModel.getMockSignature();
      delete responses[defaultResponseCode];
    }
    else {
      this.type = response.schema.type;
    }
  }

  if (errors.length > 0)
    this.resource.api.fail(errors);

  return this;
}

OperationGroup.prototype.sort = function(sorter) {

}

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
    str = 'integer'
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
  if(param['$ref'])
    str = param['$ref'];

  var schema = param.schema;
  if(schema) {
    var ref = schema['$ref'];
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
}

Operation.prototype.resolveModel = function (schema, definitions) {
  if(typeof schema['$ref'] !== 'undefined') {
    var ref = schema['$ref'];
    if(ref.indexOf('#/definitions/') == 0)
      ref = ref.substring('#/definitions/'.length);
    if(definitions[ref])
      return new Model(ref, definitions[ref]);
  }
  if(schema.type === 'array')
    return new ArrayModel(schema);
  else
    return null;
}

Operation.prototype.help = function() {
  log(this.nickname + ': ' + this.operation.summary);
  for(var i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    log('  * ' + param.name + ': ' + param.description);
  }
}

Operation.prototype.getSignature = function(type, models) {
  var isPrimitive, listType;

  if(type instanceof Array) {
    listType = true;
    type = type[0];
  }

  if(type === 'string')
    isPrimitive = true
  else
    isPrimitive = ((listType != null) && models[listType]) || (models[type] != null) ? false : true;
  if (isPrimitive) {
    return type;
  } else {
    if (listType != null)
      return models[type].getMockSignature();
    else
      return models[type].getMockSignature();
  }
};

/**
 * gets sample response for a single operation
 **/
Operation.prototype.getSampleJSON = function(type, models) {
  var isPrimitive, listType, sampleJson;

  listType = (type instanceof Array);
  isPrimitive = (models[type] != null) ? false : true;
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
}

/**
 * executes an operation
 **/
Operation.prototype.execute = function(arg1, arg2, arg3, arg4, parent) {
  var args = (arg1||{});
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

  var formParams = {};
  var headers = {};
  var requestUrl = this.path;

  success = (success||log)
  error = (error||log)

  var requiredParams = [];
  var missingParams = [];
  // check required params, track the ones that are missing
  var i;
  for(i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    if(param.required === true) {
      requiredParams.push(param.name);
      if(typeof args[param.name] === 'undefined')
        missingParams = param.name;
    }
  }

  if(missingParams.length > 0) {
    var message = 'missing required params: ' + missingParams;
    fail(message);
    return;
  }

  // set content type negotiation
  var consumes = this.consumes || this.parent.consumes || [ 'application/json' ];
  var produces = this.produces || this.parent.produces || [ 'application/json' ];

  headers = this.setContentTypes(args, opts);

  // grab params from the args, build the querystring along the way
  var querystring = "";
  for(var i = 0; i < this.parameters.length; i++) {
    var param = this.parameters[i];
    if(typeof args[param.name] !== 'undefined') {
      if(param.in === 'path') {
        var reg = new RegExp('\{' + param.name + '[^\}]*\}', 'gi');
        requestUrl = requestUrl.replace(reg, this.encodePathParam(args[param.name]));
      }
      else if (param.in === 'query' && typeof args[param.name] !== 'undefined') {
        if(querystring === '')
          querystring += '?';
        else
          querystring += '&';
        if(typeof param.collectionFormat !== 'undefined') {
          var qp = args[param.name];
          if(Array.isArray(qp))
            querystring += this.encodeCollection(param.collectionFormat, param.name, qp);
          else
            querystring += this.encodeQueryParam(param.name) + '=' + this.encodeQueryParam(args[param.name]);
        }
        else
          querystring += this.encodeQueryParam(param.name) + '=' + this.encodeQueryParam(args[param.name]);
      }
      else if (param.in === 'header')
        headers[param.name] = args[param.name];
      else if (param.in === 'formData')
        formParams[param.name] = args[param.name];
      else if (param.in === 'body')
        args.body = args[param.name];
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
    // todo append?
    args.body = encoded;
  }
  var url = this.scheme + '://' + this.host;

  if(this.basePath !== '/')
    url += this.basePath;

  url += requestUrl + querystring;

  var obj = {
    url: url,
    method: this.method,
    body: args.body,
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
  new SwaggerHttp().execute(obj);
}

Operation.prototype.setContentTypes = function(args, opts) {
  // default type
  var accepts = 'application/json';
  var consumes = 'application/json';

  var allDefinedParams = this.parameters;
  var definedFormParams = [];
  var definedFileParams = [];
  var body = args.body;
  var headers = {};

  // get params from the operation and set them in definedFileParams, definedFormParams, headers
  var i;
  for(i = 0; i < allDefinedParams.length; i++) {
    var param = allDefinedParams[i];
    if(param.in === 'formData')
      definedFormParams.push(param);
    else if(param.in === 'file')
      definedFileParams.push(param);
    else if(param.in === 'header' && this.headers) {
      var key = param.name;
      var headerValue = this.headers[param.name];
      if(typeof this.headers[param.name] !== 'undefined')
        headers[key] = headerValue;
    }
  }

  // if there's a body, need to set the accepts header via requestContentType
  if (body && (this.method === 'post' || this.method === 'put' || this.method === 'patch' || this.method === 'delete')) {
    if (opts.requestContentType)
      consumes = opts.requestContentType;
  } else {
    // if any form params, content type must be set
    if(definedFormParams.length > 0) {
      if(definedFileParams.length > 0)
        consumes = 'multipart/form-data';
      else
        consumes = 'application/x-www-form-urlencoded';
    }
    else if (this.type == 'DELETE')
      body = '{}';
    else if (this.type != 'DELETE')
      accepts = null;
  }

  if (consumes && this.consumes) {
    if (this.consumes.indexOf(consumes) === -1) {
      log('server doesn\'t consume ' + consumes + ', try ' + JSON.stringify(this.consumes));
      consumes = this.operation.consumes[0];
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
      accepts = this.produces[0];
    }
  }

  if ((consumes && body !== '') || (consumes === 'application/x-www-form-urlencoded'))
    headers['Content-Type'] = consumes;
  if (accepts)
    headers['Accept'] = accepts;
  return headers;
}

Operation.prototype.encodeCollection = function(type, name, value) {
  var encoded = '';
  var i;
  if(type === 'default' || type === 'multi') {
    for(i = 0; i < value.length; i++) {
      if(i > 0) encoded += '&'
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
    if(separator !== '') {
      for(i = 0; i < value.length; i++) {
        if(i == 0)
          encoded = this.encodeQueryParam(name) + '=' + this.encodeQueryParam(value[i]);
        else
          encoded += separator + this.encodeQueryParam(value[i]);
      }
    }
  }
  // TODO: support the different encoding schemes here
  return encoded;
}

/**
 * TODO this encoding needs to be changed 
 **/
Operation.prototype.encodeQueryParam = function(arg) {
  return escape(arg);
}

/**
 * TODO revisit, might not want to leave '/'
 **/
Operation.prototype.encodePathParam = function(pathParam) {
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

var Model = function(name, definition) {
  this.name = name;
  this.definition = definition || {};
  this.properties = [];
  var requiredFields = definition.required || [];

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
}

Model.prototype.createJSONSample = function(modelsToIgnore) {
  var result = {};
  modelsToIgnore = (modelsToIgnore||{})
  modelsToIgnore[this.name] = this;
  var i;
  for (i = 0; i < this.properties.length; i++) {
    prop = this.properties[i];
    result[prop.name] = prop.getSampleValue(modelsToIgnore);
  }
  delete modelsToIgnore[this.name];
  return result;
};

Model.prototype.getSampleValue = function(modelsToIgnore) {
  var i;
  var obj = {};
  for(i = 0; i < this.properties.length; i++ ) {
    var property = this.properties[i];
    obj[property.name] = property.sampleValue(false, modelsToIgnore);
  }
  return obj;
}

Model.prototype.getMockSignature = function(modelsToIgnore) {
  var propertiesStr = [];
  var i;
  for (i = 0; i < this.properties.length; i++) {
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
    modelsToIgnore = {};

  modelsToIgnore[this.name] = this;
  var i;
  for (i = 0; i < this.properties.length; i++) {
    var prop = this.properties[i];
    var ref = prop['$ref'];
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
  if(obj['$ref'])
    this['$ref'] = simpleRef(obj['$ref']);
  else if (obj.type === 'array') {
    if(obj.items['$ref'])
      this['$ref'] = simpleRef(obj.items['$ref']);
    else
      obj = obj.items;
  }
  this.name = name;
  this.description = obj.description;
  this.obj = obj;
  this.optional = true;
  this.default = obj.default || null;
  this.example = obj.example || null;
}

Property.prototype.getSampleValue = function (modelsToIgnore) {
  return this.sampleValue(false, modelsToIgnore);
}

Property.prototype.isArray = function () {
  var schema = this.schema;
  if(schema.type === 'array')
    return true;
  else
    return false;
}

Property.prototype.sampleValue = function(isArray, ignoredModels) {
  isArray = (isArray || this.isArray());
  ignoredModels = (ignoredModels || {});
  var type = getStringSignature(this.obj);
  var output;

  if(this['$ref']) {
    var refModelName = simpleRef(this['$ref']);
    var refModel = models[refModelName];
    if(refModel && typeof ignoredModels[type] === 'undefined') {
      ignoredModels[type] = this;
      output = refModel.getSampleValue(ignoredModels);
    }
    else
      type = refModel;
  }
  else if(this.example)
    output = this.example;
  else if(this.default)
    output = this.default;
  else if(type === 'date-time')
    output = new Date().toISOString();
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
}

getStringSignature = function(obj) {
  var str = '';
  if(obj.type === 'array') {
    obj = (obj.items || obj['$ref'] || {});
    str += 'Array[';
  }
  if(obj.type === 'integer' && obj.format === 'int32')
    str += 'integer';
  else if(obj.type === 'integer' && obj.format === 'int64')
    str += 'long';
  else if(obj.type === 'integer' && typeof obj.format === 'undefined')
    str += 'long';
  else if(obj.type === 'string' && obj.format === 'date-time')
    str += 'date-time';
  else if(obj.type === 'string' && obj.format === 'date')
    str += 'date';
  else if(obj.type === 'number' && obj.format === 'float')
    str += 'float';
  else if(obj.type === 'number' && obj.format === 'double')
    str += 'double';
  else if(obj.type === 'number' && typeof obj.format === 'undefined')
    str += 'double';
  else if(obj.type === 'boolean')
    str += 'boolean';
  else if(obj['$ref'])
    str += simpleRef(obj['$ref']);
  else
    str += obj.type;
  if(obj.type === 'array')
    str += ']';
  return str;
}

simpleRef = function(name) {
  if(typeof name === 'undefined')
    return null;
  if(name.indexOf("#/definitions/") === 0)
    return name.substring('#/definitions/'.length)
  else
    return name;
}

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
  return str;
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
}

var e = (typeof window !== 'undefined' ? window : exports);

var sampleModels = {};
var cookies = {};
var models = {};

e.authorizations = new SwaggerAuthorizations();
e.ApiKeyAuthorization = ApiKeyAuthorization;
e.PasswordAuthorization = PasswordAuthorization;
e.CookieAuthorization = CookieAuthorization;
e.SwaggerClient = SwaggerClient;

/**
 * SwaggerHttp is a wrapper for executing requests
 */
var SwaggerHttp = function() {};

SwaggerHttp.prototype.execute = function(obj) {
  if(obj && (typeof obj.useJQuery === 'boolean'))
    this.useJQuery = obj.useJQuery;
  else
    this.useJQuery = this.isIE8();

  if(this.useJQuery)
    return new JQueryHttpClient().execute(obj);
  else
    return new ShredHttpClient().execute(obj);
}

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
}

JQueryHttpClient.prototype.execute = function(obj) {
  var cb = obj.on;
  var request = obj;

  obj.type = obj.method;
  obj.cache = false;

  obj.beforeSend = function(xhr) {
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

    var contentType = (headers["content-type"]||headers["Content-Type"]||null)

    if(contentType != null) {
      if(contentType.indexOf("application/json") == 0 || contentType.indexOf("+json") > 0) {
        if(response.responseText && response.responseText !== "") {
          try {
            out.obj = JSON.parse(response.content.data);
          }
          catch (ex) {
            // do not set out.obj
            log ("unable to parse JSON content");
          }
        }
        else
          out.obj = {}
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
}

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
}

ShredHttpClient.prototype.initShred = function () {
  this.isInitialized = true;
  this.registerProcessors(this.shred);
}

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
}

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

    var contentType = (response._headers["content-type"]||response._headers["Content-Type"]||null)

    if(contentType != null) {
      if(contentType.indexOf("application/json") == 0 || contentType.indexOf("+json") > 0) {
        if(response.content.data && response.content.data !== "")
          try{
            out.obj = JSON.parse(response.content.data);
          }
          catch (e) {
            // unable to parse
          }
        else
          out.obj = {}
      }
    }
    return out;
  };

  res = {
    error: function(response) {
      if (obj)
        return cb.error(transform(response));
    },
    redirect: function(response) {
      if (obj)
        return cb.redirect(transform(response));
    },
    307: function(response) {
      if (obj)
        return cb.redirect(transform(response));
    },
    response: function(response) {
      if (obj)
        return cb.response(transform(response));
    }
  };
  if (obj) {
    obj.on = res;
  }
  return this.shred.request(obj);
};