target_link_libraries(${PROJECT_NAME_TEST} PUBLIC
    ${PROJECT_NAME}
    gtest
)
target_sources(${PROJECT_NAME_TEST} PUBLIC
    test_main.cpp
)
