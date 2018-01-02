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
import Foundation

// A class for loading Stencil templates from compiled-in representations

public class TemplateLoader: Loader {
  private var templates: [String:String]

  public init() {
    self.templates = loadTemplates()
  }

  public func loadTemplate(name: String, environment: Environment) throws -> Template {
    if let encoding = templates[name],
      let data = Data(base64Encoded: encoding, options:[]),
      let template = String(data:data, encoding:.utf8) {
      return environment.templateClass.init(templateString: template,
                                            environment: environment,
                                            name: name)
    } else {
      throw TemplateDoesNotExist(templateNames: [name], loader: self)
    }
  }

  public func loadTemplate(names: [String], environment: Environment) throws -> Template {
    for name in names {
      if let encoding = templates[name],
        let data = Data(base64Encoded: encoding, options:[]),
        let template = String(data:data, encoding:.utf8) {
        return environment.templateClass.init(templateString: template,
                                              environment: environment,
                                              name: name)
      }
    }
    throw TemplateDoesNotExist(templateNames: names, loader: self)
  }
}
