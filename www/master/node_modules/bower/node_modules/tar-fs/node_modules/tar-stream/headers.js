var ZEROS = '0000000000000000000'
var ZERO_OFFSET = '0'.charCodeAt(0)
var USTAR = 'ustar\x0000'

var clamp = function(index, len, defaultValue) {
  if (typeof index !== 'number') return defaultValue
  index = ~~index  // Coerce to integer.
  if (index >= len) return len
  if (index >= 0) return index
  index += len
  if (index >= 0) return index
  return 0
}

var toType = function(flag) {
  switch (flag) {
    case 0:
    return 'file'
    case 1:
    return 'link'
    case 2:
    return 'symlink'
    case 3:
    return 'character-device'
    case 4:
    return 'block-device'
    case 5:
    return 'directory'
    case 6:
    return 'fifo'
    case 7:
    return 'contiguous-file'
    case 72:
    return 'pax-header'
    case 55:
    return 'pax-global-header'
    case 27:
    return 'gnu-long-link-path'
    case 28:
    case 30:
    return 'gnu-long-path'
  }

  return null
}

var toTypeflag = function(flag) {
  switch (flag) {
    case 'file':
    return 0
    case 'link':
    return 1
    case 'symlink':
    return 2
    case 'character-device':
    return 3
    case 'block-device':
    return 4
    case 'directory':
    return 5
    case 'fifo':
    return 6
    case 'contiguous-file':
    return 7
    case 'pax-header':
    return 72
  }

  return 0
}

var alloc = function(size) {
  var buf = new Buffer(size)
  buf.fill(0)
  return buf
}

var indexOf = function(block, num, offset, end) {
  for (; offset < end; offset++) {
    if (block[offset] === num) return offset
  }
  return end
}

var cksum = function(block) {
  var sum = 8 * 32
  for (var i = 0; i < 148; i++)   sum += block[i]
  for (var i = 156; i < 512; i++) sum += block[i]
  return sum
}

var encodeOct = function(val, n) {
  val = val.toString(8)
  return ZEROS.slice(0, n-val.length)+val+' '
}

var decodeOct = function(val, offset) {
  // Older versions of tar can prefix with spaces
  while (offset < val.length && val[offset] === 32) offset++
  var end = clamp(indexOf(val, 32, offset, val.length), val.length, val.length)
  while (offset < end && val[offset] === 0) offset++
  if (end === offset) return 0
  return parseInt(val.slice(offset, end).toString(), 8)
}

var decodeStr = function(val, offset, length) {
  return val.slice(offset, indexOf(val, 0, offset, offset+length)).toString();
}

var addLength = function(str) {
  var len = Buffer.byteLength(str)
  var digits = Math.floor(Math.log(len) / Math.log(10)) + 1
  if (len + digits > Math.pow(10, digits)) digits++

  return (len+digits)+str
}

exports.decodeLongPath = function(buf) {
  return decodeStr(buf, 0, buf.length)
}

exports.encodePax = function(opts) { // TODO: encode more stuff in pax
  var result = ''
  if (opts.name) result += addLength(' path='+opts.name+'\n')
  if (opts.linkname) result += addLength(' linkpath='+opts.linkname+'\n')
  return new Buffer(result)
}

exports.decodePax = function(buf) {
  var result = {}

  while (buf.length) {
    var i = 0
    while (i < buf.length && buf[i] !== 32) i++

    var len = parseInt(buf.slice(0, i).toString())
    if (!len) return result

    var b = buf.slice(i+1, len-1).toString()
    var keyIndex = b.indexOf('=')
    if (keyIndex === -1) return result
    result[b.slice(0, keyIndex)] = b.slice(keyIndex+1)

    buf = buf.slice(len)
  }

  return result
}

exports.encode = function(opts) {
  var buf = alloc(512)
  var name = opts.name
  var prefix = ''

  if (opts.typeflag === 5 && name[name.length-1] !== '/') name += '/'
  if (Buffer.byteLength(name) !== name.length) return null // utf-8

  while (Buffer.byteLength(name) > 100) {
    var i = name.indexOf('/')
    if (i === -1) return null
    prefix += prefix ? '/' + name.slice(0, i) : name.slice(0, i)
    name = name.slice(i+1)
  }

  if (Buffer.byteLength(name) > 100 || Buffer.byteLength(prefix) > 155) return null
  if (opts.linkname && Buffer.byteLength(opts.linkname) > 100) return null

  buf.write(name)
  buf.write(encodeOct(opts.mode & 07777, 6), 100)
  buf.write(encodeOct(opts.uid, 6), 108)
  buf.write(encodeOct(opts.gid, 6), 116)
  buf.write(encodeOct(opts.size, 11), 124)
  buf.write(encodeOct((opts.mtime.getTime() / 1000) | 0, 11), 136)

  buf[156] = ZERO_OFFSET + toTypeflag(opts.type)

  if (opts.linkname) buf.write(opts.linkname, 157)

  buf.write(USTAR, 257)
  if (opts.uname) buf.write(opts.uname, 265)
  if (opts.gname) buf.write(opts.gname, 297)
  buf.write(encodeOct(opts.devmajor || 0, 6), 329)
  buf.write(encodeOct(opts.devminor || 0, 6), 337)

  if (prefix) buf.write(prefix, 345)

  buf.write(encodeOct(cksum(buf), 6), 148)

  return buf
}

exports.decode = function(buf) {
  var typeflag = buf[156] === 0 ? 0 : buf[156] - ZERO_OFFSET
  var type = toType(typeflag)

  var name = decodeStr(buf, 0, 100)
  var mode = decodeOct(buf, 100)
  var uid = decodeOct(buf, 108)
  var gid = decodeOct(buf, 116)
  var size = decodeOct(buf, 124)
  var mtime = decodeOct(buf, 136)
  var linkname = buf[157] === 0 ? null : decodeStr(buf, 157, 100)
  var uname = decodeStr(buf, 265, 32)
  var gname = decodeStr(buf, 297, 32)
  var devmajor = decodeOct(buf, 329)
  var devminor = decodeOct(buf, 337)

  if (buf[345]) name = decodeStr(buf, 345, 155)+'/'+name

  var c = cksum(buf)

  //checksum is still initial value if header was null.
  if (c === 8*32) return null

  //valid checksum
  if (c !== decodeOct(buf, 148)) throw new Error('invalid header')

  return {
    name: name,
    mode: mode,
    uid: uid,
    gid: gid,
    size: size,
    mtime: new Date(1000 * mtime),
    type: toType(typeflag),
    linkname: linkname,
    uname: uname,
    gname: gname,
    devmajor: devmajor,
    devminor: devminor
  }
}
