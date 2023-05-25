function eventListener(event) {
    event.preventDefault();
    const fields = ["fixed_acidity", 
                    "volatile_acidity", 
                    "citric_acid", 
                    "residual_sugar", 
                    "chlorides",
                    'density',
                    'ph',
                    'sulphates',
                    "free_sulfur_dioxide",
                    'alcohol',
                    'total_sulfur_dioxide'];
    const emptyFields = fields.filter(field => {
        const input = document.querySelector(`#${field}`);
        return !input.value;
    });
    if (emptyFields.length > 0) {
        event.preventDefault();
        alert('All fields are required');
        return;
    }
    form.submit();
}
