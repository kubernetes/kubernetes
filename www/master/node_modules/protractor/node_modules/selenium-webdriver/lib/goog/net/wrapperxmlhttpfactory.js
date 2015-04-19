// Copyright 2010 The Closure Library Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Implementation of XmlHttpFactory which allows construction from
 * simple factory methods.
 * @author dbk@google.com (David Barrett-Kahn)
 */

goog.provide('goog.net.WrapperXmlHttpFactory');

/** @suppress {extraRequire} Typedef. */
goog.require('goog.net.XhrLike');
goog.require('goog.net.XmlHttpFactory');



/**
 * An xhr factory subclass which can be constructed using two factory methods.
 * This exists partly to allow the preservation of goog.net.XmlHttp.setFactory()
 * with an unchanged signature.
 * @param {function():!goog.net.XhrLike.OrNative} xhrFactory
 *     A function which returns a new XHR object.
 * @param {function():!Object} optionsFactory A function which returns the
 *     options associated with xhr objects from this factory.
 * @extends {goog.net.XmlHttpFactory}
 * @constructor
 * @final
 */
goog.net.WrapperXmlHttpFactory = function(xhrFactory, optionsFactory) {
  goog.net.XmlHttpFactory.call(this);

  /**
   * XHR factory method.
   * @type {function() : !goog.net.XhrLike.OrNative}
   * @private
   */
  this.xhrFactory_ = xhrFactory;

  /**
   * Options factory method.
   * @type {function() : !Object}
   * @private
   */
  this.optionsFactory_ = optionsFactory;
};
goog.inherits(goog.net.WrapperXmlHttpFactory, goog.net.XmlHttpFactory);


/** @override */
goog.net.WrapperXmlHttpFactory.prototype.createInstance = function() {
  return this.xhrFactory_();
};


/** @override */
goog.net.WrapperXmlHttpFactory.prototype.getOptions = function() {
  return this.optionsFactory_();
};

