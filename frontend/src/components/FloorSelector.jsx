/**
 * FloorSelector supports active floor selection for future multi-floor navigation.
 *
 * @param {object} props
 * @param {Array<string|{label:string,value:string}>} props.floors - Floor options.
 * @param {string} props.value - Current selected floor value.
 * @param {(floorValue:string)=>void} props.onChange - Change callback.
 */
export default function FloorSelector({ floors, value, onChange }) {
  const options = Array.isArray(floors)
    ? floors.map((floor) => {
        if (typeof floor === "string") {
          return { label: floor, value: floor };
        }
        return {
          label: String(floor?.label ?? floor?.value ?? "Floor"),
          value: String(floor?.value ?? floor?.label ?? "Floor"),
        };
      })
    : [];

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
        {options.map((floor) => (
          <option key={floor.value} value={floor.value}>
            {floor.label}
          </option>
        ))}
      </select>
      <div className="muted floor-note">Vertical connectors and floor transitions are ready for expansion.</div>
    </section>
  );
}
