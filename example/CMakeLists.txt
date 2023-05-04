set(src
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)
add_executable(distance_transform_example ${src})

target_link_libraries(distance_transform_example distance_transform)
