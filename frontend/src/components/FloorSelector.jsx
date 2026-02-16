/**
 * FloorSelector supports active floor selection for future multi-floor navigation.
 *
 * @param {object} props
 * @param {string[]} props.floors - List of floor labels.
 * @param {string} props.value - Current floor.
 * @param {(floor:string)=>void} props.onChange - Change callback.
 */
export default function FloorSelector({ floors, value, onChange }) {
  return (
    <section className="panel floor-panel" aria-label="Floor selector">
      <h2>Floor</h2>
      <label htmlFor="floor-select" className="muted">
        Active Floor
      </label>
      <select
        id="floor-select"
        className="select"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-label="Select floor"
      >
        {floors.map((floor) => (
          <option key={floor} value={floor}>
            {floor}
          </option>
        ))}
      </select>
    </section>
  );
}
