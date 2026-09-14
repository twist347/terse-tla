add_executable(tla_test main.cpp)
target_link_libraries(tla_test PRIVATE tla::tla)
target_compile_options(tla_test PRIVATE ${TLA_WARNINGS})
add_test(NAME tla_test COMMAND tla_test)
