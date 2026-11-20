'use strict'

const SVG_NS = 'http://www.w3.org/2000/svg'

function svgnew(tag, attributes) {
  const element = document.createElementNS(SVG_NS, tag)
  svgattr(element, attributes)
  return element
}

function svgattr(element, attributes) {
  if (attributes) {
    for (const k in attributes) {
      if (Object.hasOwn(attributes, k)) {
        element.setAttributeNS(null, k, attributes[k])
      }
    }
  }
}

function svgattach(parent, child) {
  // eslint-disable-next-line unicorn/prefer-dom-node-append
  return parent.appendChild(child)
}

function svgadd(element, tag, attributes) {
  return svgattach(element, svgnew(tag, attributes))
}

function newArray(n, function_) {
  const array = Array.from({length: n})
  for (let i = 0; i < n; i++) {
    array[i] = function_(i)
  }

  return array
}

function arrayEq(a, b) {
  if (a === b) {
    return true
  }

  if (a === undefined || a === null || b === undefined || b === null) {
    return false
  }

  if (a.length !== b.length) {
    return false
  }

  for (const [i, element] of a.entries()) {
    if (element !== b[i]) {
      return false
    }
  }

  return true
}

function formatCallReturn(callTime, returnTime) {
  return '<br><br>Call: ' + callTime + '<br><br>Return: ' + returnTime
}

// eslint-disable-next-line no-unused-vars, complexity
function render(data) {
  const PADDING = 10
  const BOX_HEIGHT = 30
  const BOX_SPACE = 15
  const EPSILON = 20
  const LINE_BLEED = 5
  const BOX_GAP = 20
  const BOX_TEXT_PADDING = 10
  const HISTORY_RECT_RADIUS = 4

  const annotations = data.Annotations
  const coreHistory = data.Partitions
  // For simplicity, make annotations look like more history
  const allData = [...coreHistory, {History: annotations}]

  let maxClient = -1
  for (const partition of allData) {
    for (const element of partition.History) {
      maxClient = Math.max(maxClient, element.ClientId)
    }
  }

  // "real" clients, not including tags
  const realClients = maxClient + 1
  // We treat each unique annotation tag as another "client"
  const tags = new Set()
  for (const annot of annotations) {
    const tag = annot.Tag
    if (tag.length > 0) {
      tags.add(tag)
    }
  }

  // Add synthetic client numbers
  const tag2ClientId = {}
  // `tags` is a collection of strings, so we do want to sort lexicographically
  // eslint-disable-next-line unicorn/require-array-sort-compare
  const sortedTags = [...tags].toSorted()
  for (const tag of sortedTags) {
    maxClient += 1
    tag2ClientId[tag] = maxClient
  }

  for (const annot of annotations) {
    const tag = annot.Tag
    if (tag.length > 0) {
      annot.ClientId = tag2ClientId[tag]
    }
  }

  // Total number of clients now includes these synthetic clients
  const nClient = maxClient + 1

  // Prepare some useful data to be used later:
  // - Add a GID to each event
  // - Create a mapping from GIDs back to events
  // - Create a set of all timestamps
  // - Create a set of all start timestamps
  const allTimestamps = new Set()
  const startTimestamps = new Set()
  const endTimestamps = new Set()
  let gid = 0
  const byGid = {}
  for (const partition of allData) {
    for (const element of partition.History) {
      allTimestamps.add(element.Start)
      startTimestamps.add(element.Start)
      endTimestamps.add(element.End)
      allTimestamps.add(element.End)
      // Give elements GIDs
      element.Gid = gid
      byGid[gid] = element
      gid++
    }
  }

  let sortedTimestamps = [...allTimestamps].toSorted((a, b) => a - b)

  // If one event has the same end time as another's start time, that means that
  // they are concurrent, and we need to display them with overlap. We do this
  // by tweaking the events that share the end time, updating the time to
  // end+epsilon, so we have overlap.
  //
  // We do not render a good visualization in the situation where a single
  // client has two events where one has an end time that matches the other's
  // start time.  There isn't an easy way to handle this, because these two
  // operations are concurrent (see the comment in model.go for more details for
  // why it must be this way), and we can't display them with overlap on the
  // same row cleanly.
  const epsilon = 16
  // Coordinated with computeVisualizationData, which uses an incrementing
  // counter multiplied by 100, so it's safe to adjust timestamps by += epsilon
  // without it overlapping with another adjusted timestamp (2*16 < 100)
  for (const [index, partition] of allData.entries()) {
    if (index === allData.length - 1) {
      continue // Last partition is the annotations
    }

    for (const element of partition.History) {
      const end = element.End
      if (startTimestamps.has(end)) {
        element.End = end + epsilon
        allTimestamps.add(element.End)
      }
    }
  }

  // Handle display of (1) annotations where one has the same end time as
  // another's start time, and (2) point-in-time annotations, on the same row.
  //
  // Unlike operations, where we interpret the start and end times as a closed
  // interval (and where they have to be displayed with overlap, to be able to
  // render linearizations and partial linearizations correctly), annotations
  // are for display purposes only and we can interpret annotations with
  // different start/end times as open intervals, and render two annotations
  // with times (a, b) and (b, c) without overlap. We can also handle the
  // situation where we have a point-in-time annotation with times (b, b).
  //
  // This code currently does not handle the case where we have two
  // point-in-time annotations with the same tag at the same timestamp.
  //
  // We keep the annotation adjustment epsilon even smaller (dividing by 2), so
  // adjusting an event's end time forward by epsilon doesn't overlap with an
  // annotation's start time that's adjusted forwards (the adjustment of
  // annotations goes in the opposite direction as that for events).
  for (const element of allData.at(-1).History) {
    if (element.End === element.Start) {
      // Point-in-time annotation: we adjust these to have a non-zero-duration;
      // we only need to edit the end timestamp, and we can leave the start
      // as-is
      element.End += epsilon / 4
      allTimestamps.add(element.End)
    } else {
      // Annotation touching another event or annotation
      if (startTimestamps.has(element.End)) {
        element.End -= epsilon / 2
        allTimestamps.add(element.End)
      }

      if (endTimestamps.has(element.Start)) {
        element.Start += epsilon / 2
        allTimestamps.add(element.Start)
      }
    }
  }

  // Update sortedTimestamps, because we created some new timestamps.
  sortedTimestamps = [...allTimestamps].toSorted((a, b) => a - b)

  // Compute layout.
  //
  // We warp time to make it easier to see what's going on. We can think
  // of there being a monotonically increasing mapping from timestamps to
  // x-positions. This mapping should satisfy some criteria to make the
  // visualization interpretable:
  //
  // - distinguishability: there should be some minimum distance between
  // unequal timestamps
  // - visible text: history boxes should be wide enough to fit the text
  // they contain
  // - enough space for LPs: history boxes should be wide enough to fit
  // all linearization points that go through them, while maintaining
  // readability of linearizations (where each LP in a sequence is spaced
  // some minimum distance away from the previous one)
  //
  // Originally, I thought about this as a linear program:
  //
  // - variables for every unique timestamp, x_i = warp(timestamp_i)
  // - objective: minimize sum x_i
  // - constraint: non-negative
  // - constraint: ordering + distinguishability, timestamp_i < timestamp_j -> x_i + EPS < x_j
  // - constraint: visible text, size_text_j < x_{timestamp_j_end} - x_{timestamp_j_start}
  // - constraint: linearization lines have points that fit within box, ...
  //
  // This used to actually be implemented using an LP solver (without the
  // linearization point part, though that should be doable too), but
  // then I realized it's possible to solve optimally using a greedy
  // left-to-right scan in linear time.
  //
  // So that is what we do here. We optimally solve the above, and while
  // doing so, also compute some useful information (e.g. x-positions of
  // linearization points) that is useful later.
  const xPos = {}
  // Compute some information about history elements, sorted by end time;
  // the most important information here is box width.
  const byEnd = allData
    .flatMap((partition) =>
      partition.History.map((element) => {
        // Compute width of the text inside the history element by actually
        // drawing it (in a hidden div)
        const scratch = document.querySelector('#calc')
        scratch.innerHTML = ''
        const svg = svgadd(scratch, 'svg')
        const text = svgadd(svg, 'text', {
          'text-anchor': 'middle',
          class: 'history-text',
        })
        text.textContent = element.Description
        const bbox = text.getBBox()
        const width = bbox.width + 2 * BOX_TEXT_PADDING
        return {
          start: element.Start,
          end: element.End,
          width,
          gid: element.Gid,
        }
      }),
    )
    .toSorted((a, b) => a.end - b.end)
  // Some preprocessing for linearization points and illegal next
  // linearizations. We need to figure out where exactly LPs end up
  // as we go, so we can make sure event boxes are wide enough.
  const eventToLinearizations = newArray(gid, () => []) // Event -> [{index, position}]
  const eventIllegalLast = newArray(gid, () => []) // Event -> [index]
  const allLinearizations = []
  let lgid = 0
  for (const partition of coreHistory) {
    for (const lin of partition.PartialLinearizations) {
      const globalized = [] // Linearization with global indexes instead of partition-local ones
      const included = new Set() // For figuring out illegal next LPs
      for (const [position, id] of lin.entries()) {
        included.add(id.Index)
        const eventGid = partition.History[id.Index].Gid
        globalized.push(eventGid)
        eventToLinearizations[eventGid].push({index: lgid, position})
      }

      allLinearizations.push(globalized)
      let minEnd = Infinity
      for (const [index, element] of partition.History.entries()) {
        if (!included.has(index)) {
          minEnd = Math.min(minEnd, element.End)
        }
      }

      for (const [index, element] of partition.History.entries()) {
        if (!included.has(index) && element.Start < minEnd) {
          eventIllegalLast[element.Gid].push(lgid)
        }
      }

      lgid++
    }
  }

  const linearizationPositions = newArray(lgid, () => []) // [[xpos]]
  // Okay, now we're ready to do the left-to-right scan.
  // Solve timestamp -> xPos.
  let eventIndex = 0
  xPos[sortedTimestamps[0]] = 0 // Positions start at 0
  for (let i = 1; i < sortedTimestamps.length; i++) {
    // Left-to-right scan, finding minimum time we can use
    const ts = sortedTimestamps[i]
    // Ensure some gap from last timestamp
    let pos = xPos[sortedTimestamps[i - 1]] + BOX_GAP
    // Ensure that text fits in boxes
    while (eventIndex < byEnd.length && byEnd[eventIndex].end <= ts) {
      // Push our position as far as necessary to accommodate text in box
      const event_ = byEnd[eventIndex]
      const textEndPos = xPos[event_.start] + event_.width
      pos = Math.max(pos, textEndPos)
      // Ensure that LPs fit in box.
      //
      // When placing the end of an event, for all partial linearizations
      // that include that event, for the prefix that comes before that event,
      // all their start points must have been placed already, so we can figure
      // out the minimum width that the box needs to be to accommodate the LP.
      for (const li of [
        ...eventToLinearizations[event_.gid],
        ...eventIllegalLast[event_.gid].map((index) => ({
          index,
          position: allLinearizations[index].length - 1,
        })),
      ]) {
        const {index, position} = li
        for (let j = linearizationPositions[index].length; j <= position; j++) {
          // Determine past points
          const previous =
            linearizationPositions[index].length > 0 ? linearizationPositions[index][j - 1] : null

          const nextGid = allLinearizations[index][j]
          const nextPos =
            previous === null
              ? xPos[byGid[nextGid].Start]
              : Math.max(xPos[byGid[nextGid].Start], previous + EPSILON)

          linearizationPositions[index].push(nextPos)
        }

        // This next line only really makes sense for the ones in
        // eventToLinearizations, not the ones from eventIllegalLast,
        // but it's safe to do it for all points, so we don't bother to
        // distinguish.
        pos = Math.max(pos, linearizationPositions[index][position])
      }

      // Ensure that illegal next LPs fit in box too
      for (const li of eventIllegalLast[event_.gid]) {
        const lin = linearizationPositions[li]
        const previous = lin.at(-1)
        pos = Math.max(pos, previous + EPSILON)
      }

      eventIndex++
    }

    xPos[ts] = pos
  }

  // Get maximum tag width
  let maxTagWidth = 0
  for (let i = 0; i < nClient; i++) {
    const tag = i < realClients ? i.toString() : sortedTags[i - realClients]
    const scratch = document.querySelector('#calc')
    scratch.innerHTML = ''
    const svg = svgadd(scratch, 'svg')
    const text = svgadd(svg, 'text', {
      'text-anchor': 'end',
    })
    text.textContent = tag
    const bbox = text.getBBox()
    const width = bbox.width + 2 * BOX_TEXT_PADDING
    if (width > maxTagWidth) {
      maxTagWidth = width
    }
  }

  const t0x = PADDING + maxTagWidth // X-pos of line at t=0

  // Solved, now draw UI.

  let selected = false
  let selectedIndex = [-1, -1]

  const canvasHeight = 2 * PADDING + BOX_HEIGHT * nClient + BOX_SPACE * (nClient - 1)
  const canvasWidth = 2 * PADDING + maxTagWidth + xPos[sortedTimestamps.at(-1)]
  const svg = svgadd(document.querySelector('#canvas'), 'svg', {
    height: canvasHeight,
    width: canvasWidth,
  })

  // Draw background, etc.
  const bg = svgadd(svg, 'g')
  const bgRect = svgadd(bg, 'rect', {
    height: canvasHeight,
    width: canvasWidth,
    x: 0,
    y: 0,
    class: 'bg',
  })
  bgRect.addEventListener('click', handleBgClick)
  for (let i = 0; i < nClient; i++) {
    const text = svgadd(bg, 'text', {
      x: PADDING + maxTagWidth - BOX_TEXT_PADDING,
      y: PADDING + BOX_HEIGHT / 2 + i * (BOX_HEIGHT + BOX_SPACE),
      'text-anchor': 'end',
    })
    text.textContent = i < realClients ? i : sortedTags[i - realClients]
  }

  // Vertical line at t=0
  svgadd(bg, 'line', {
    x1: t0x,
    y1: PADDING,
    x2: t0x,
    y2: canvasHeight - PADDING,
    class: 'divider',
  })
  // Horizontal line dividing clients from annotation tags, but only if there are tags
  if (tags.size > 0) {
    const annotationLineY = PADDING + realClients * (BOX_HEIGHT + BOX_SPACE) - BOX_SPACE / 2
    svgadd(bg, 'line', {
      x1: PADDING,
      y1: annotationLineY,
      x2: t0x,
      y2: annotationLineY,
      class: 'divider',
    })
  }

  // Draw history
  const historyLayers = []
  const historyRects = []
  const targetRects = svgnew('g')
  for (const [partitionIndex, partition] of allData.entries()) {
    const l = svgadd(svg, 'g')
    historyLayers.push(l)
    const rects = []
    for (const [elementIndex, element] of partition.History.entries()) {
      const g = svgadd(l, 'g')
      const rx = xPos[element.Start]
      const width = xPos[element.End] - rx
      const x = rx + t0x
      const y = PADDING + element.ClientId * (BOX_HEIGHT + BOX_SPACE)
      const rectClass = element.Annotation ? 'client-annotation-rect' : 'history-rect'
      rects.push(
        svgadd(g, 'rect', {
          height: BOX_HEIGHT,
          width,
          x,
          y,
          rx: HISTORY_RECT_RADIUS,
          ry: HISTORY_RECT_RADIUS,
          class: rectClass,
          style:
            element.Annotation && element.BackgroundColor.length > 0
              ? `fill: ${element.BackgroundColor};`
              : '',
        }),
      )
      const text = svgadd(g, 'text', {
        x: x + width / 2,
        y: y + BOX_HEIGHT / 2,
        'text-anchor': 'middle',
        class: 'history-text',
        style:
          element.Annotation && element.TextColor.length > 0 ? `fill: ${element.TextColor};` : '',
      })
      text.textContent = element.Description
      // We don't add mouseTarget to g, but to targetRects, because we
      // want to layer this on top of everything at the end; otherwise, the
      // LPs and lines will be over the target, which will create holes
      // where hover etc. won't work
      const mouseTarget = svgadd(targetRects, 'rect', {
        height: BOX_HEIGHT,
        width,
        x,
        y,
        class: 'target-rect',
        'data-partition': partitionIndex,
        'data-index': elementIndex,
      })
      mouseTarget.addEventListener('mouseover', handleMouseOver)
      mouseTarget.addEventListener('mousemove', handleMouseMove)
      mouseTarget.addEventListener('mouseout', handleMouseOut)
      mouseTarget.addEventListener('click', handleClick)
    }

    historyRects.push(rects)
  }

  // Draw partial linearizations
  const illegalLast = coreHistory.map((partition) =>
    partition.PartialLinearizations.map(() => new Set()),
  )
  const largestIllegal = coreHistory.map(() => ({}))
  const largestIllegalLength = coreHistory.map(() => ({}))
  const partialLayers = []
  const errorPoints = []
  for (const [partitionIndex, partition] of coreHistory.entries()) {
    const l = []
    partialLayers.push(l)
    for (const [linIndex, lin] of partition.PartialLinearizations.entries()) {
      const g = svgadd(svg, 'g')
      l.push(g)
      let previousX = null
      let previousY = null
      let previousElement = null
      const included = new Set()
      for (const id of lin) {
        const element = partition.History[id.Index]
        const hereX = t0x + xPos[element.Start]
        const x = previousX === null ? hereX : Math.max(hereX, previousX + EPSILON)
        const y = PADDING + element.ClientId * (BOX_HEIGHT + BOX_SPACE) - LINE_BLEED
        // Line from previous
        if (previousElement !== null) {
          svgadd(g, 'line', {
            x1: previousX,
            x2: x,
            y1:
              previousElement.ClientId >= element.ClientId
                ? previousY
                : previousY + BOX_HEIGHT + 2 * LINE_BLEED,
            y2: previousElement.ClientId <= element.ClientId ? y : y + BOX_HEIGHT + 2 * LINE_BLEED,
            class: 'linearization linearization-line',
          })
        }

        // Current line
        svgadd(g, 'line', {
          x1: x,
          x2: x,
          y1: y,
          y2: y + BOX_HEIGHT + 2 * LINE_BLEED,
          class: 'linearization linearization-point',
        })
        previousX = x
        previousY = y
        previousElement = element
        included.add(id.Index)
      }

      // Show possible but illegal next linearizations
      // a history element is a possible next try
      // if no other history element must be linearized earlier
      // i.e. forall others, this.start < other.end
      let minEnd = Infinity
      for (const [index, element] of partition.History.entries()) {
        if (!included.has(index)) {
          minEnd = Math.min(minEnd, element.End)
        }
      }

      for (const [index, element] of partition.History.entries()) {
        if (!included.has(index) && element.Start < minEnd) {
          const hereX = t0x + xPos[element.Start]
          const x = previousX === null ? hereX : Math.max(hereX, previousX + EPSILON)
          const y = PADDING + element.ClientId * (BOX_HEIGHT + BOX_SPACE) - LINE_BLEED
          // Line from previous
          svgadd(g, 'line', {
            x1: previousX,
            x2: x,
            y1:
              previousElement.ClientId >= element.ClientId
                ? previousY
                : previousY + BOX_HEIGHT + 2 * LINE_BLEED,
            y2: previousElement.ClientId <= element.ClientId ? y : y + BOX_HEIGHT + 2 * LINE_BLEED,
            class: 'linearization-invalid linearization-line',
          })
          // Current line
          const point = svgadd(g, 'line', {
            x1: x,
            x2: x,
            y1: y,
            y2: y + BOX_HEIGHT + 2 * LINE_BLEED,
            class: 'linearization-invalid linearization-point',
          })
          errorPoints.push({
            x,
            partition: partitionIndex,
            index: lin.at(-1).Index, // NOTE not index
            element: point,
          })
          illegalLast[partitionIndex][linIndex].add(index)
          // eslint-disable-next-line max-depth
          if (
            !Object.hasOwn(largestIllegalLength[partitionIndex], index) ||
            largestIllegalLength[partitionIndex][index] < lin.length
          ) {
            largestIllegalLength[partitionIndex][index] = lin.length
            largestIllegal[partitionIndex][index] = linIndex
          }
        }
      }
    }
  }

  errorPoints.sort((a, b) => a.x - b.x)

  // Attach targetRects
  svgattach(svg, targetRects)

  // Tooltip
  // eslint-disable-next-line unicorn/prefer-dom-node-append
  const tooltip = document.querySelector('#canvas').appendChild(document.createElement('div'))
  tooltip.setAttribute('class', 'tooltip')

  function handleMouseOver() {
    if (selected) {
      return
    }

    const partition = Number(this.dataset.partition)
    const index = Number(this.dataset.index)
    highlight(partition, index)
    tooltip.style.display = 'block'
  }

  function linearizationIndex(partition, index) {
    // Show this linearization
    if (partition >= coreHistory.length) {
      // Annotation
      return null
    }

    if (Object.hasOwn(coreHistory[partition].Largest, index)) {
      return coreHistory[partition].Largest[index]
    }

    if (Object.hasOwn(largestIllegal[partition], index)) {
      return largestIllegal[partition][index]
    }

    return null
  }

  function highlight(partition, index) {
    // Hide all but this partition
    for (const [i, layer] of historyLayers.entries()) {
      layer.classList.toggle('hidden', i !== partition)
    }

    // Hide all but the relevant linearization
    for (const layer of partialLayers) {
      for (const g of layer) {
        g.classList.add('hidden')
      }
    }

    // Show this linearization
    const maxIndex = linearizationIndex(partition, index)
    if (maxIndex !== null) {
      partialLayers[partition][maxIndex].classList.remove('hidden')
    }

    updateJump()
  }

  let lastTooltip = [null, null, null, null, null]
  function handleMouseMove(event_) {
    // Keep tooltip static if selected
    if (selected) {
      return
    }

    const partition = Number(this.dataset.partition)
    const index = Number(this.dataset.index)
    const [sPartition, sIndex] = selectedIndex
    const thisTooltip = [partition, index, selected, sPartition, sIndex]

    if (!arrayEq(lastTooltip, thisTooltip)) {
      // If selected, show info relevant to the selected linearization
      const maxIndex = selected
        ? linearizationIndex(sPartition, sIndex)
        : linearizationIndex(partition, index)

      const callTime = allData[partition].History[index].OriginalStart
      const returnTime = allData[partition].History[index].OriginalEnd

      let metadata = ''
      if (partition < coreHistory.length) {
        const m = allData[partition].History[index].Metadata
        if (m !== '') {
          metadata = m + '<br><br>'
        }
      }

      if (partition >= coreHistory.length) {
        // Annotation
        const details = annotations[index].Details
        tooltip.innerHTML = details.length === 0 ? '&langle;no details&rangle;' : details
      } else if (selected && sPartition !== partition) {
        tooltip.innerHTML =
          metadata + 'Not part of selected partition.' + formatCallReturn(callTime, returnTime)
      } else if (maxIndex === null) {
        tooltip.innerHTML =
          metadata +
          (selected
            ? 'Selected element is not part of any partial linearization.'
            : 'Not part of any partial linearization.') +
          formatCallReturn(callTime, returnTime)
      } else {
        const lin = coreHistory[partition].PartialLinearizations[maxIndex]
        let previous = null
        let current = null
        let found = false
        for (const element of lin) {
          previous = current
          current = element
          if (current.Index === index) {
            found = true
            break
          }
        }

        let message = metadata

        if (found) {
          // Part of linearization
          if (previous !== null) {
            message +=
              '<strong>Previous state:</strong><br>' + previous.StateDescription + '<br><br>'
          }

          message +=
            '<strong>New state:</strong><br>' +
            current.StateDescription +
            formatCallReturn(callTime, returnTime)
        } else if (illegalLast[partition][maxIndex].has(index)) {
          // Illegal next one
          message +=
            '<strong>Previous state:</strong><br>' +
            lin.at(-1).StateDescription +
            '<br><br><strong>New state:</strong><br>&langle;invalid op&rangle;' +
            formatCallReturn(callTime, returnTime)
        } else {
          // Not part of this one
          message +=
            "Not part of selected element's partial linearization." +
            formatCallReturn(callTime, returnTime)
        }

        tooltip.innerHTML = message
      }

      lastTooltip = thisTooltip
    }

    // Make sure tooltip doesn't overflow off the right side of the screen
    const maxX =
      document.documentElement.scrollLeft +
      document.documentElement.clientWidth -
      PADDING -
      tooltip.getBoundingClientRect().width
    tooltip.style.left = Math.min(event_.pageX + 20, maxX) + 'px'
    tooltip.style.top = event_.pageY + 20 + 'px'
  }

  function handleMouseOut() {
    if (selected) {
      return
    }

    resetHighlight()
    tooltip.style.display = 'none'
    lastTooltip = [null, null, null, null, null]
  }

  function resetHighlight() {
    // Show all layers
    for (const layer of historyLayers) {
      layer.classList.remove('hidden')
    }

    // Show longest linearizations, which are first
    for (const layers of partialLayers) {
      for (const [i, l] of layers.entries()) {
        l.classList.toggle('hidden', i !== 0)
      }
    }

    updateJump()
  }

  // Store jump click handler so we can remove it before adding a new one
  let jumpClickHandler = null

  function updateJump() {
    const jump = document.querySelector('#jump-link')
    // Find first non-hidden point
    // feels a little hacky, but it works
    const point = errorPoints.find((pt) => !pt.element.parentElement.classList.contains('hidden'))

    // Remove any existing event listener
    if (jumpClickHandler) {
      jump.removeEventListener('click', jumpClickHandler)
      jumpClickHandler = null
    }

    if (point) {
      jump.classList.remove('inactive')
      jumpClickHandler = () => {
        point.element.scrollIntoView({behavior: 'smooth', inline: 'center', block: 'center'})
        if (!selected) {
          select(point.partition, point.index)
        }
      }

      jump.addEventListener('click', jumpClickHandler)
    } else {
      jump.classList.add('inactive')
    }
  }

  function handleClick(event_) {
    const partition = Number(this.dataset.partition)
    const index = Number(this.dataset.index)
    if (selected) {
      const [sPartition, sIndex] = selectedIndex
      if (partition === sPartition && index === sIndex) {
        deselect()
        // Note: we're still displaying the tooltip, but once the user's mouse moves, it'll get updated
        return
      }

      historyRects[sPartition][sIndex].classList.remove('selected')
    }

    select(partition, index)

    tooltip.style.display = 'block'
    // Set static tooltip position when selecting
    const maxX =
      document.documentElement.scrollLeft +
      document.documentElement.clientWidth -
      PADDING -
      tooltip.getBoundingClientRect().width
    tooltip.style.left = Math.min(event_.pageX + 20, maxX) + 'px'
    tooltip.style.top = event_.pageY + 20 + 'px'
  }

  function handleBgClick() {
    deselect()

    tooltip.style.display = 'none'
    lastTooltip = [null, null, null, null, null]
  }

  function select(partition, index) {
    selected = true
    selectedIndex = [partition, index]
    highlight(partition, index)
    historyRects[partition][index].classList.add('selected')
  }

  function deselect() {
    if (!selected) {
      return
    }

    selected = false
    resetHighlight()
    const [partition, index] = selectedIndex
    historyRects[partition][index].classList.remove('selected')
  }

  handleMouseOut() // Initialize, same as mouse out
}
