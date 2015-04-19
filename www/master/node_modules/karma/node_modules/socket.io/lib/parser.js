
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

/**
 * Packet types.
 */

var packets = exports.packets = {
      'disconnect': 0
    , 'connect': 1
    , 'heartbeat': 2
    , 'message': 3
    , 'json': 4
    , 'event': 5
    , 'ack': 6
    , 'error': 7
    , 'noop': 8
  }
  , packetslist = Object.keys(packets);

/**
 * Errors reasons.
 */

var reasons = exports.reasons = {
      'transport not supported': 0
    , 'client not handshaken': 1
    , 'unauthorized': 2
  }
  , reasonslist = Object.keys(reasons);

/**
 * Errors advice.
 */

var advice = exports.advice = {
      'reconnect': 0
  }
  , advicelist = Object.keys(advice);

/**
 * Encodes a packet.
 *
 * @api private
 */

exports.encodePacket = function (packet) {
  var type = packets[packet.type]
    , id = packet.id || ''
    , endpoint = packet.endpoint || ''
    , ack = packet.ack
    , data = null;

  switch (packet.type) {
    case 'message':
      if (packet.data !== '')
        data = packet.data;
      break;

    case 'event':
      var ev = { name: packet.name };

      if (packet.args && packet.args.length) {
        ev.args = packet.args;
      }

      data = JSON.stringify(ev);
      break;

    case 'json':
      data = JSON.stringify(packet.data);
      break;

    case 'ack':
      data = packet.ackId
        + (packet.args && packet.args.length
            ? '+' + JSON.stringify(packet.args) : '');
      break;

    case 'connect':
      if (packet.qs)
        data = packet.qs;
      break;

     case 'error':
      var reason = packet.reason ? reasons[packet.reason] : ''
        , adv = packet.advice ? advice[packet.advice] : ''

      if (reason !== '' || adv !== '')
        data = reason + (adv !== '' ? ('+' + adv) : '')

      break;
  }

  // construct packet with required fragments
  var encoded = type + ':' + id + (ack == 'data' ? '+' : '') + ':' + endpoint;

  // data fragment is optional
  if (data !== null && data !== undefined)
    encoded += ':' + data;

  return encoded;
};

/**
 * Encodes multiple messages (payload).
 *
 * @param {Array} messages
 * @api private
 */

exports.encodePayload = function (packets) {
  var decoded = '';

  if (packets.length == 1)
    return packets[0];

  for (var i = 0, l = packets.length; i < l; i++) {
    var packet = packets[i];
    decoded += '\ufffd' + packet.length + '\ufffd' + packets[i]
  }

  return decoded;
};

/**
 * Decodes a packet
 *
 * @api private
 */

var regexp = /([^:]+):([0-9]+)?(\+)?:([^:]+)?:?([\s\S]*)?/;

/**
 * Wrap the JSON.parse in a seperate function the crankshaft optimizer will
 * only punish this function for the usage for try catch
 *
 * @api private
 */

function parse (data) {
  try { return JSON.parse(data) }
  catch (e) { return false }
}

exports.decodePacket = function (data) {
  var pieces = data.match(regexp);

  if (!pieces) return {};

  var id = pieces[2] || ''
    , data = pieces[5] || ''
    , packet = {
          type: packetslist[pieces[1]]
        , endpoint: pieces[4] || ''
      };

  // whether we need to acknowledge the packet
  if (id) {
    packet.id = id;
    if (pieces[3])
      packet.ack = 'data';
    else
      packet.ack = true;
  }

  // handle different packet types
  switch (packet.type) {
    case 'message':
      packet.data = data || '';
      break;

    case 'event':
      pieces = parse(data);
      if (pieces) {
        packet.name = pieces.name;
        packet.args = pieces.args;
      }

      packet.args = packet.args || [];
      break;

    case 'json':
      packet.data = parse(data);
      break;

    case 'connect':
      packet.qs = data || '';
      break;

    case 'ack':
      pieces = data.match(/^([0-9]+)(\+)?(.*)/);
      if (pieces) {
        packet.ackId = pieces[1];
        packet.args = [];

        if (pieces[3]) {
          packet.args = parse(pieces[3]) || [];
        }
      }
      break;

    case 'error':
      pieces = data.split('+');
      packet.reason = reasonslist[pieces[0]] || '';
      packet.advice = advicelist[pieces[1]] || '';
  }

  return packet;
};

/**
 * Decodes data payload. Detects multiple messages
 *
 * @return {Array} messages
 * @api public
 */

exports.decodePayload = function (data) {
  if (undefined == data || null == data) {
    return [];
  }

  if (data[0] == '\ufffd') {
    var ret = [];

    for (var i = 1, length = ''; i < data.length; i++) {
      if (data[i] == '\ufffd') {
        ret.push(exports.decodePacket(data.substr(i + 1, length)));
        i += Number(length) + 1;
        length = '';
      } else {
        length += data[i];
      }
    }

    return ret;
  } else {
    return [exports.decodePacket(data)];
  }
};
