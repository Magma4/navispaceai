import { useMemo } from "react";

/**
 * RoomSelector provides room-based start/goal endpoint selection.
 *
 * Expected room shape:
 * {
 *   room_id: string,
 *   floor_number: number,
 *   name: string,
 *   centroid_m: [x, y, z]
 * }
 *
 * @param {object} props
 * @param {Array<object>} props.rooms
 * @param {number|null} props.activeFloor
 * @param {string} props.startRoomId
 * @param {string} props.goalRoomId
 * @param {(roomId:string)=>void} props.onStartRoomChange
 * @param {(roomId:string)=>void} props.onGoalRoomChange
 * @param {boolean} [props.disabled=false]
 */
export default function RoomSelector({
  rooms,
  activeFloor,
  startRoomId,
  goalRoomId,
  onStartRoomChange,
  onGoalRoomChange,
  disabled = false,
}) {
  const floorRooms = useMemo(() => {
    if (!Array.isArray(rooms)) return [];
    if (activeFloor === null || activeFloor === undefined) return rooms;
    return rooms.filter((room) => Number(room.floor_number) === Number(activeFloor));
  }, [rooms, activeFloor]);

  return (
    <section className="panel room-selector" aria-label="Room selector">
      <h2>Room Endpoints</h2>
      <p className="muted">Select start and goal by indexed room metadata.</p>

      <label>
        <span className="muted">Start Room</span>
        <select
          className="select"
          value={startRoomId}
          onChange={(event) => onStartRoomChange(event.target.value)}
          disabled={disabled}
        >
          <option value="">Select start room</option>
          {floorRooms.map((room) => (
            <option key={room.room_id} value={room.room_id}>
              {room.name} ({room.room_id})
            </option>
          ))}
        </select>
      </label>

      <label>
        <span className="muted">Goal Room</span>
        <select
          className="select"
          value={goalRoomId}
          onChange={(event) => onGoalRoomChange(event.target.value)}
          disabled={disabled}
        >
          <option value="">Select goal room</option>
          {floorRooms.map((room) => (
            <option key={room.room_id} value={room.room_id}>
              {room.name} ({room.room_id})
            </option>
          ))}
        </select>
      </label>
    </section>
  );
}
