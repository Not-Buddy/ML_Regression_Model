"""
Sample data for testing the house price prediction system
"""

def get_sample_feature_list():
    """Return predefined sample features for testing"""
    return [
        (2, 1100, 2),   # 2 BHK, 1100 sqft, 2 bathrooms
        (3, 1600, 3),   # 3 BHK, 1600 sqft, 3 bathrooms
        (1, 600, 1),    # 1 BHK, 600 sqft, 1 bathroom
        (4, 2200, 4),   # 4 BHK, 2200 sqft, 4 bathrooms
        (2, 900, 2),    # 2 BHK, 900 sqft, 2 bathrooms
    ]

def get_test_cases():
    """Return test cases with expected behavior"""
    return {
        'valid_cases': [
            {'input': (2, 1000, 2), 'description': 'Standard 2BHK apartment'},
            {'input': (3, 1500, 3), 'description': 'Spacious 3BHK house'},
            {'input': (1, 500, 1), 'description': 'Compact 1BHK unit'},
        ],
        'edge_cases': [
            {'input': (1, 300, 1), 'description': 'Minimum size property'},
            {'input': (4, 3000, 4), 'description': 'Large family house'},
        ]
    }
