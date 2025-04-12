import '../pages/home.css';

function SearchBar({ value, onChange, onSearch }) {

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
        onSearch(event.target.value.toUpperCase()); 
      }
  };

  return (
    <input
      type="text"
      placeholder="Ticker"
      value={value}
      onChange={onChange}
      onKeyDown={handleKeyDown}
    />
  );
}

export default SearchBar;