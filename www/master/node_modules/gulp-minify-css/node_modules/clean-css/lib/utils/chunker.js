// Divides `data` into chunks of `chunkSize` for faster processing
function Chunker(data, breakString, chunkSize) {
  this.chunks = [];

  for (var cursor = 0, dataSize = data.length; cursor < dataSize;) {
    var nextCursor = cursor + chunkSize > dataSize ?
      dataSize - 1 :
      cursor + chunkSize;

    if (data[nextCursor] != breakString)
      nextCursor = data.indexOf(breakString, nextCursor);
    if (nextCursor == -1)
      nextCursor = data.length - 1;

    this.chunks.push(data.substring(cursor, nextCursor + breakString.length));
    cursor = nextCursor + breakString.length;
  }
}

Chunker.prototype.isEmpty = function () {
  return this.chunks.length === 0;
};

Chunker.prototype.next = function () {
  return this.chunks.shift();
};

module.exports = Chunker;
