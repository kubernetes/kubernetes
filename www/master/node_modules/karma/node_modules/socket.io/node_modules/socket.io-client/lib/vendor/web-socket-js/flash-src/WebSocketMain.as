// Copyright: Hiroshi Ichikawa <http://gimite.net/en/>
// License: New BSD License
// Reference: http://dev.w3.org/html5/websockets/
// Reference: http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol-76

package {

import flash.display.Sprite;
import flash.external.ExternalInterface;
import flash.system.Security;
import flash.utils.setTimeout;

import mx.utils.URLUtil;

/**
  * Provides JavaScript API of WebSocket.
  */
public class WebSocketMain extends Sprite implements IWebSocketLogger{
  
  private var callerUrl:String;
  private var debug:Boolean = false;
  private var manualPolicyFileLoaded:Boolean = false;
  private var webSockets:Array = [];
  private var eventQueue:Array = [];
  
  public function WebSocketMain() {
    ExternalInterface.addCallback("setCallerUrl", setCallerUrl);
    ExternalInterface.addCallback("setDebug", setDebug);
    ExternalInterface.addCallback("create", create);
    ExternalInterface.addCallback("send", send);
    ExternalInterface.addCallback("close", close);
    ExternalInterface.addCallback("loadManualPolicyFile", loadManualPolicyFile);
    ExternalInterface.addCallback("receiveEvents", receiveEvents);
    ExternalInterface.call("WebSocket.__onFlashInitialized");
  }
  
  public function setCallerUrl(url:String):void {
    callerUrl = url;
  }
  
  public function setDebug(val:Boolean):void {
    debug = val;
  }
  
  private function loadDefaultPolicyFile(wsUrl:String):void {
    var policyUrl:String = "xmlsocket://" + URLUtil.getServerName(wsUrl) + ":843";
    log("policy file: " + policyUrl);
    Security.loadPolicyFile(policyUrl);
  }
  
  public function loadManualPolicyFile(policyUrl:String):void {
    log("policy file: " + policyUrl);
    Security.loadPolicyFile(policyUrl);
    manualPolicyFileLoaded = true;
  }
  
  public function log(message:String):void {
    if (debug) {
      ExternalInterface.call("WebSocket.__log", encodeURIComponent("[WebSocket] " + message));
    }
  }
  
  public function error(message:String):void {
    ExternalInterface.call("WebSocket.__error", encodeURIComponent("[WebSocket] " + message));
  }
  
  private function parseEvent(event:WebSocketEvent):Object {
    var webSocket:WebSocket = event.target as WebSocket;
    var eventObj:Object = {};
    eventObj.type = event.type;
    eventObj.webSocketId = webSocket.getId();
    eventObj.readyState = webSocket.getReadyState();
    eventObj.protocol = webSocket.getAcceptedProtocol();
    if (event.message !== null) {
      eventObj.message = event.message;
    }
    return eventObj;
  }
  
  public function create(
      webSocketId:int,
      url:String, protocols:Array,
      proxyHost:String = null, proxyPort:int = 0,
      headers:String = null):void {
    if (!manualPolicyFileLoaded) {
      loadDefaultPolicyFile(url);
    }
    var newSocket:WebSocket = new WebSocket(
        webSocketId, url, protocols, getOrigin(), proxyHost, proxyPort,
        getCookie(url), headers, this);
    newSocket.addEventListener("open", onSocketEvent);
    newSocket.addEventListener("close", onSocketEvent);
    newSocket.addEventListener("error", onSocketEvent);
    newSocket.addEventListener("message", onSocketEvent);
    webSockets[webSocketId] = newSocket;
  }
  
  public function send(webSocketId:int, encData:String):int {
    var webSocket:WebSocket = webSockets[webSocketId];
    return webSocket.send(encData);
  }
  
  public function close(webSocketId:int):void {
    var webSocket:WebSocket = webSockets[webSocketId];
    webSocket.close();
  }
  
  public function receiveEvents():Object {
    var result:Object = eventQueue;
    eventQueue = [];
    return result;
  }
  
  private function getOrigin():String {
    return (URLUtil.getProtocol(this.callerUrl) + "://" +
      URLUtil.getServerNameWithPort(this.callerUrl)).toLowerCase();
  }
  
  private function getCookie(url:String):String {
    if (URLUtil.getServerName(url).toLowerCase() ==
        URLUtil.getServerName(this.callerUrl).toLowerCase()) {
      return ExternalInterface.call("function(){return document.cookie}");
    } else {
      return "";
    }
  }
  
  /**
   * Socket event handler.
   */
  public function onSocketEvent(event:WebSocketEvent):void {
    var eventObj:Object = parseEvent(event);
    eventQueue.push(eventObj);
    processEvents();
  }
  
  /**
   * Process our event queue.  If javascript is unresponsive, set
   * a timeout and try again.
   */
  public function processEvents():void {
    if (eventQueue.length == 0) return;
    if (!ExternalInterface.call("WebSocket.__onFlashEvent")) {
      setTimeout(processEvents, 500);
    }
  }
  
}

}
