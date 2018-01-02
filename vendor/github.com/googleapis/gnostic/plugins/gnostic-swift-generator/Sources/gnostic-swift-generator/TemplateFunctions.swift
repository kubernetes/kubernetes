// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Stencil

func TemplateExtensions() -> Extension {
  let ext = Extension()

  ext.registerFilter("hasParameters") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    return method.parametersType != nil
  }
  ext.registerFilter("hasResponses") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    return method.responsesType != nil
  }
  ext.registerFilter("syncClientParametersDeclaration") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    var result = ""
    if let parametersType = method.parametersType {
      for field in parametersType.fields {
        if result != "" {
          result += ", "
        }
        result += field.name + " : " + field.typeName
      }
    }
    return result
  }
  ext.registerFilter("syncClientReturnDeclaration") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    var result = ""
    if let resultTypeName = method.resultTypeName {
      result = " -> " + resultTypeName
    }
    return result
  }
  ext.registerFilter("asyncClientParametersDeclaration") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    var result = ""
    if let parametersType = method.parametersType {
      for field in parametersType.fields {
        if result != "" {
          result += ", "
        }
        result += field.name + " : " + field.typeName
      }
    }
    // add callback
    if result != "" {
      result += ", "
    }
    if let resultTypeName = method.resultTypeName {
      result += "callback : @escaping (" + resultTypeName + "?, Swift.Error?)->()"
    } else {
      result += "callback : @escaping (Swift.Error?)->()"
    }
    return result
  }
  ext.registerFilter("protocolParametersDeclaration") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    var result = ""
    if let parametersTypeName = method.parametersTypeName {
      result = "_ parameters : " + parametersTypeName
    }
    return result
  }
  ext.registerFilter("protocolReturnDeclaration") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    var result = ""
    if let responsesTypeName = method.responsesTypeName {
      result = "-> " + responsesTypeName
    }
    return result
  }
  ext.registerFilter("parameterFieldNames") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    var result = ""
    if let parametersType = method.parametersType {
      for field in parametersType.fields {
        if result != "" {
          result += ", "
        }
        result += field.name + ":" + field.name
      }
    }
    return result
  }
  ext.registerFilter("parametersTypeFields") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    if let parametersType = method.parametersType {
      return parametersType.fields
    } else {
      return []
    }
  }
  ext.registerFilter("kituraPath") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    var path = method.path
    if let parametersType = method.parametersType {
      for field in parametersType.fields {
        if field.position == "path" {
          let original = "{" + field.jsonName + "}"
          let replacement = ":" + field.jsonName
          path = path.replacingOccurrences(of:original, with:replacement)
        }
      }
    }
    return path
  }
  ext.registerFilter("bodyParameterFieldName") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    if let parametersType = method.parametersType {
      for field in parametersType.fields {
        if field.position == "body" {
          return field.name
        }
      }
    }
    return ""
  }
  ext.registerFilter("responsesHasFieldNamedOK") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    if let responsesType = method.responsesType {
      for field in responsesType.fields {
        if field.name == "ok" {
          return true
        }
      }
    }
    return false
  }
  ext.registerFilter("responsesHasFieldNamedError") { (value: Any?, arguments: [Any?]) in
    let method : ServiceMethod = value as! ServiceMethod
    if let responsesType = method.responsesType {
      for field in responsesType.fields {
        if field.name == "error" {
          return true
        }
      }
    }
    return false
  }

  return ext
}
