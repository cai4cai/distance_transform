set(src
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)
add_executable(DopeVector_example ${src})

target_link_libraries(DopeVector_example DopeVector)
