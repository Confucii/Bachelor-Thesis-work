## Questions
Why cannot we use predefined joints instead of segment.
### Like:
```
joint1 = anchor_point + np.array([0, 1]) * length_joint
joint2 = joint1 + np.array([0, 1]) * length_joint
joint3 = joint2 + np.array([0, 1]) * length_joint
instead of 
segment = np.array([0, 1]) * length_joint
```

### Why  calculate rotation of the second segment
``` 
joint2 = np.dot(rotation(theta_1), (segment + np.dot(rotation(theta_2), segment)))
this works and this doesn’t
rotation1 = np.dot(rotation(theta_1), rotation(theta_2))
joint2 = np.dot(rotation1, joint2)
```
## Task 2
### Inverse Kinematics
#### Derivatives
![Image](../Images/Derivatives.png "Inverse Kinematics")
#### 3 segment arm
![Image](../Images/Inverse.png "3 segment arm")
![Image](../Images/Inverse_2.png "3 segment arm")

### Regression
![Image](../Images/Regression.png "Regression 1")
### Backpropagation
![Image](../Images/Backpropagation.png "Regression 2")

